import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ReadResourceRequestSchema,
  CallToolResult,
  ListToolsResult,
  ListResourcesResult,
  ReadResourceResult,
  Tool,
  Resource,
  ProgressTokenSchema,
  TextContent,
  CallToolRequest,
  ReadResourceRequest,
} from '@modelcontextprotocol/sdk/types.js';
import { createWorker, Worker } from 'tesseract.js';
import * as path from 'path';
import { PDFExtract } from 'pdf.js-extract';
import mammoth from 'mammoth';

// Interfaces para nuestros tipos de datos
interface TranslationResult {
  originalText: string;
  translatedText: string;
  confidence: 'high' | 'medium' | 'low';
  notes: string[];
  terminology: TerminologyMatch[];
}

interface TerminologyMatch {
  term: string;
  translation: string;
  context: string;
  source: string;
}

interface GlossaryEntry {
  original: string;
  translation: string;
  context: string;
  source: string;
  period: string;
}

interface GlossaryStats {
  name: string;
  entryCount: number;
  lastModified: string;
}

interface TranslationStats {
  totalTranslations: number;
  glossariesLoaded: number;
  serverUptime: number;
}

// Tipos para argumentos de herramientas
interface ExtractTextArgs {
  filePath: string;
  documentType?: 'facsimile' | 'glossary' | 'reference';
}

interface LoadGlossaryArgs {
  filePath: string;
  glossaryName: string;
}

interface TranslateTextArgs {
  text: string;
  context?: string;
  useGlossaries?: string[];
}

interface SearchTerminologyArgs {
  term: string;
  contextFilter?: string;
}

class HistoricalTranslationServer {
  private server: Server;
  private glossaries: Map<string, GlossaryEntry[]> = new Map();
  private tessWorker: Worker | null = null;
  private translationCount = 0;
  private startTime = Date.now();

  constructor() {
    this.server = new Server(
      {
        name: 'historical-translation-server',
        version: '1.0.0',
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  private setupHandlers(): void {
    // Manejo de herramientas
    this.server.setRequestHandler(ListToolsRequestSchema, async (): Promise<ListToolsResult> => {
      const tools: Tool[] = [
        {
          name: 'extract_text_from_document',
          description: 'Extrae texto de documentos PNG, PDF usando OCR especializado',
          inputSchema: {
            type: 'object',
            properties: {
              filePath: {
                type: 'string',
                description: 'Ruta al archivo PNG o PDF',
              },
              documentType: {
                type: 'string',
                enum: ['facsimile', 'glossary', 'reference'],
                description: 'Tipo de documento para optimizar el OCR',
              },
            },
            required: ['filePath'],
          },
        },
        {
          name: 'load_glossary',
          description: 'Carga un glosario desde archivo .doc o .pdf',
          inputSchema: {
            type: 'object',
            properties: {
              filePath: {
                type: 'string',
                description: 'Ruta al archivo del glosario',
              },
              glossaryName: {
                type: 'string',
                description: 'Nombre identificador del glosario',
              },
            },
            required: ['filePath', 'glossaryName'],
          },
        },
        {
          name: 'translate_text',
          description: 'Traduce texto del inglés moderno temprano al español contemporáneo',
          inputSchema: {
            type: 'object',
            properties: {
              text: {
                type: 'string',
                description: 'Texto a traducir',
              },
              context: {
                type: 'string',
                description: 'Contexto histórico/teológico del texto',
              },
              useGlossaries: {
                type: 'array',
                items: { type: 'string' },
                description: 'Nombres de glosarios a usar en la traducción',
              },
              progressToken: {
                type: ['string', 'number'],
                description: 'Token para notificaciones de progreso',
                optional: true,
              },
            },
            required: ['text'],
          },
        },
        {
          name: 'search_terminology',
          description: 'Busca terminología específica en los glosarios cargados',
          inputSchema: {
            type: 'object',
            properties: {
              term: {
                type: 'string',
                description: 'Término a buscar',
              },
              contextFilter: {
                type: 'string',
                description: 'Filtro de contexto (teológico, histórico, etc.)',
              },
            },
            required: ['term'],
          },
        },
      ];

      return { tools };
    });

    // Manejo de recursos
    this.server.setRequestHandler(ListResourcesRequestSchema, async (): Promise<ListResourcesResult> => {
      const resources: Resource[] = [
        {
          uri: 'glossaries://loaded',
          mimeType: 'application/json',
          name: 'Glosarios cargados',
          description: 'Lista de todos los glosarios disponibles',
        },
        {
          uri: 'stats://translation',
          mimeType: 'application/json',
          name: 'Estadísticas de traducción',
          description: 'Estadísticas de uso y rendimiento',
        },
      ];

      return { resources };
    });

    // Lectura de recursos
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request: ReadResourceRequest): Promise<ReadResourceResult> => {
      const { uri } = request.params;

      if (uri === 'glossaries://loaded') {
        const glossaryList: GlossaryStats[] = Array.from(this.glossaries.entries()).map(([name, entries]) => ({
          name,
          entryCount: entries.length,
          lastModified: new Date().toISOString(),
        }));
        
        const content: TextContent = {
          type: 'text',
          text: JSON.stringify(glossaryList, null, 2),
        };

        return {
          contents: [content],
        };
      }

      if (uri === 'stats://translation') {
        const stats: TranslationStats = {
          totalTranslations: this.translationCount,
          glossariesLoaded: this.glossaries.size,
          serverUptime: Math.floor((Date.now() - this.startTime) / 1000),
        };

        const content: TextContent = {
          type: 'text',
          text: JSON.stringify(stats, null, 2),
        };

        return {
          contents: [content],
        };
      }

      throw new McpError(ErrorCode.InvalidRequest, `Recurso no encontrado: ${uri}`);
    });

    // Ejecución de herramientas
    this.server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest): Promise<CallToolResult> => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'extract_text_from_document':
            return await this.extractTextFromDocument(args as ExtractTextArgs);
          
          case 'load_glossary':
            return await this.loadGlossary(args as LoadGlossaryArgs);
          
          case 'translate_text':
            return await this.translateText(args as TranslateTextArgs);
          
          case 'search_terminology':
            return await this.searchTerminology(args as SearchTerminologyArgs);
          
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Herramienta no encontrada: ${name}`);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Error desconocido';
        throw new McpError(ErrorCode.InternalError, `Error ejecutando ${name}: ${errorMessage}`);
      }
    });
  }

  private async initTesseract(): Promise<void> {
    if (!this.tessWorker) {
      this.tessWorker = await createWorker();
      await this.tessWorker.loadLanguage('eng');
      await this.tessWorker.initialize('eng');
      // Configuración específica para textos históricos
      await this.tessWorker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}"\'-/\\',
        tessedit_pageseg_mode: '1', // Automatic page segmentation with OSD
        preserve_interword_spaces: '1',
      });
    }
  }

  private async extractTextFromDocument(args: ExtractTextArgs): Promise<CallToolResult> {
    try {
      const { filePath, documentType } = args;
      const fileExt = path.extname(filePath).toLowerCase();
      
      if (fileExt === '.pdf') {
        return await this.extractTextFromPDF(filePath);
      } else if (['.png', '.jpg', '.jpeg', '.tiff'].includes(fileExt)) {
        return await this.extractTextFromImage(filePath, documentType);
      } else {
        throw new Error(`Formato de archivo no soportado: ${fileExt}`);
      }
    } catch (error) {
      const content: TextContent = {
        type: 'text',
        text: `Error extrayendo texto: ${error instanceof Error ? error.message : 'Error desconocido'}`,
      };

      return {
        content: [content],
      };
    }
  }

  private async extractTextFromPDF(filePath: string): Promise<CallToolResult> {
    try {
      const pdfExtract = new PDFExtract();
      const data = await pdfExtract.extract(filePath, {});
      
      const text = data.pages.map(page => 
        page.content.map(item => item.str).join(' ')
      ).join('\n\n');

      const content: TextContent = {
        type: 'text',
        text: `Texto extraído del PDF (${data.pages.length} páginas):\n\n${text}`,
      };

      return {
        content: [content],
      };
    } catch (error) {
      const content: TextContent = {
        type: 'text',
        text: `Error extrayendo texto del PDF: ${error instanceof Error ? error.message : 'Error desconocido'}`,
      };

      return {
        content: [content],
      };
    }
  }

  private async extractTextFromImage(filePath: string, documentType?: string): Promise<CallToolResult> {
    try {
      await this.initTesseract();
      
      if (!this.tessWorker) {
        throw new Error('No se pudo inicializar Tesseract');
      }

      const { data: { text } } = await this.tessWorker.recognize(filePath);
      
      // Post-procesamiento básico para textos históricos
      const cleanedText = this.cleanHistoricalText(text);
      
      const content: TextContent = {
        type: 'text',
        text: `Texto extraído de la imagen:\n\n${cleanedText}`,
      };

      return {
        content: [content],
      };
    } catch (error) {
      const content: TextContent = {
        type: 'text',
        text: `Error extrayendo texto de la imagen: ${error instanceof Error ? error.message : 'Error desconocido'}`,
      };

      return {
        content: [content],
      };
    }
  }

  private cleanHistoricalText(text: string): string {
    // Correcciones básicas para OCR de textos históricos
    return text
      .replace(/\s+/g, ' ') // Espacios múltiples
      .replace(/([a-z])([A-Z])/g, '$1 $2') // Separar palabras pegadas
      .replace(/[ſ]/g, 's') // Reemplazar s larga
      .replace(/[ﬀ]/g, 'ff') // Ligaduras
      .replace(/[ﬁ]/g, 'fi')
      .replace(/[ﬂ]/g, 'fl')
      .trim();
  }

  private async loadGlossary(args: LoadGlossaryArgs): Promise<CallToolResult> {
    try {
      const { filePath, glossaryName } = args;
      const fileExt = path.extname(filePath).toLowerCase();
      let text = '';

      if (fileExt === '.doc' || fileExt === '.docx') {
        const result = await mammoth.extractRawText({ path: filePath });
        text = result.value;
      } else if (fileExt === '.pdf') {
        const pdfExtract = new PDFExtract();
        const data = await pdfExtract.extract(filePath, {});
        text = data.pages.map(page => 
          page.content.map(item => item.str).join(' ')
        ).join('\n');
      } else {
        throw new Error(`Formato de glosario no soportado: ${fileExt}`);
      }

      // Parsear el glosario (implementación básica)
      const entries = this.parseGlossaryText(text);
      this.glossaries.set(glossaryName, entries);

      const content: TextContent = {
        type: 'text',
        text: `Glosario '${glossaryName}' cargado exitosamente. ${entries.length} entradas procesadas.`,
      };

      return {
        content: [content],
      };
    } catch (error) {
      const content: TextContent = {
        type: 'text',
        text: `Error cargando glosario: ${error instanceof Error ? error.message : 'Error desconocido'}`,
      };

      return {
        content: [content],
      };
    }
  }

  private parseGlossaryText(text: string): GlossaryEntry[] {
    // Implementación básica - se mejorará en fases posteriores
    const entries: GlossaryEntry[] = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
      // Buscar patrones como "término: traducción" o "término - traducción"
      const match = line.match(/^([^:-]+)[:|-]\s*(.+)$/);
      if (match) {
        entries.push({
          original: match[1].trim(),
          translation: match[2].trim(),
          context: '',
          source: 'imported',
          period: '17th century',
        });
      }
    }
    
    return entries;
  }

  private async translateText(args: TranslateTextArgs): Promise<CallToolResult> {
    try {
      const { text, context, useGlossaries, progressToken } = args;
      
      // Implementación básica - se expandirá en fases posteriores
      const result: TranslationResult = {
        originalText: text,
        translatedText: text, // Por ahora, devolvemos el texto original
        confidence: 'low',
        notes: ['Traducción básica - requiere revisión humana'],
        terminology: [],
      };

      if (progressToken) {
        // Simulate some progress
        this.server.notify('notifications/progress', { progressToken, progress: 10, total: 100 });
      }

      // Buscar términos en glosarios
      if (useGlossaries && useGlossaries.length > 0) {
        for (const glossaryName of useGlossaries) {
          const glossary = this.glossaries.get(glossaryName);
          if (glossary) {
            for (const entry of glossary) {
              if (text.toLowerCase().includes(entry.original.toLowerCase())) {
                result.terminology.push({
                  term: entry.original,
                  translation: entry.translation,
                  context: entry.context,
                  source: glossaryName,
                });
              }
            }
          }
          if (progressToken) {
            // Simulate progress for each glossary processed
            const currentProgress = 10 + (useGlossaries.indexOf(glossaryName) + 1) * (80 / useGlossaries.length);
            this.server.notify('notifications/progress', { progressToken, progress: currentProgress, total: 100 });
          }
        }
      }

      this.translationCount++;

      const content: TextContent = {
        type: 'text',
        text: JSON.stringify(result, null, 2),
      };

      return {
        content: [content],
      };
    } catch (error) {
      const content: TextContent = {
        type: 'text',
        text: `Error traduciendo texto: ${error instanceof Error ? error.message : 'Error desconocido'}`,
      };

      return {
        content: [content],
      };
    }
  }

  private async searchTerminology(args: SearchTerminologyArgs): Promise<CallToolResult> {
    try {
      const { term, contextFilter } = args;
      const matches: TerminologyMatch[] = [];
      
      for (const [glossaryName, entries] of this.glossaries) {
        for (const entry of entries) {
          if (entry.original.toLowerCase().includes(term.toLowerCase()) ||
              entry.translation.toLowerCase().includes(term.toLowerCase())) {
            
            // Aplicar filtro de contexto si se especifica
            if (!contextFilter || entry.context.toLowerCase().includes(contextFilter.toLowerCase())) {
              matches.push({
                term: entry.original,
                translation: entry.translation,
                context: entry.context,
                source: glossaryName,
              });
            }
          }
        }
      }

      const content: TextContent = {
        type: 'text',
        text: JSON.stringify(matches, null, 2),
      };

      return {
        content: [content],
      };
    } catch (error) {
      const content: TextContent = {
        type: 'text',
        text: `Error buscando terminología: ${error instanceof Error ? error.message : 'Error desconocido'}`,
      };

      return {
        content: [content],
      };
    }
  }

  async run(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Servidor MCP de Traducción Histórica ejecutándose en stdio');
  }

  async cleanup(): Promise<void> {
    if (this.tessWorker) {
      await this.tessWorker.terminate();
      this.tessWorker = null;
    }
  }
}

// Inicializar y ejecutar el servidor
const server = new HistoricalTranslationServer();

// Manejo de señales de terminación
process.on('SIGINT', async () => {
  console.error('Cerrando servidor...');
  await server.cleanup();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.error('Cerrando servidor...');
  await server.cleanup();
  process.exit(0);
});

server.run().catch(console.error);