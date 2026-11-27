import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  LineRuleType,
  Packer,
  PageBreak,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableLayoutType,
  TableOfContents,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} from "docx";

const FONT = {
  ascii: "Times New Roman",
  hAnsi: "Times New Roman",
  eastAsia: "Times New Roman",
  cs: "Times New Roman",
};
const MONO = { ascii: "Consolas", hAnsi: "Consolas", eastAsia: "Consolas", cs: "Consolas" };
const CONTENT_WIDTH = 9071; // 16 cm in twips
const CELL_BORDER = { style: BorderStyle.SINGLE, size: 1, color: "808080" };
const CELL_BORDERS = {
  top: CELL_BORDER,
  bottom: CELL_BORDER,
  left: CELL_BORDER,
  right: CELL_BORDER,
};

function run(text, { size = 13, bold = false, italics = false, color, mono = false } = {}) {
  return new TextRun({
    text,
    bold,
    italics,
    color,
    size: size * 2,
    font: mono ? MONO : FONT,
  });
}

function bodyParagraph(text) {
  return new Paragraph({ children: [run(text)] });
}

function heading(level, text) {
  const map = {
    1: HeadingLevel.HEADING_1,
    2: HeadingLevel.HEADING_2,
    3: HeadingLevel.HEADING_3,
  };
  return new Paragraph({ heading: map[level], children: [new TextRun(text)] });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function captionParagraph(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 120 },
    children: [run(text, { size: 11, italics: true })],
  });
}

function sectionTitle(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [run(text, { size: 14, bold: true })],
  });
}

function columnWidths(headers, rows) {
  const weights = headers.map((header, index) => {
    const cellLengths = rows.map((row) => String(row[index] ?? "").length);
    const longest = Math.max(header.length, ...cellLengths, 6);
    return Math.min(Math.max(longest, 8), 42);
  });
  const total = weights.reduce((sum, value) => sum + value, 0);
  const widths = weights.map((weight) =>
    Math.floor((weight / total) * CONTENT_WIDTH),
  );
  const drift = CONTENT_WIDTH - widths.reduce((sum, value) => sum + value, 0);
  widths[widths.length - 1] += drift;
  return widths;
}

function buildTable(block) {
  const headers = block.headers;
  const rows = block.rows;
  const widths = columnWidths(headers, rows);
  const alignments = block.align ?? headers.map(() => "center");

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map(
      (header, index) =>
        new TableCell({
          borders: CELL_BORDERS,
          width: { size: widths[index], type: WidthType.DXA },
          shading: { fill: "D9D9D9", type: ShadingType.CLEAR },
          verticalAlign: VerticalAlign.CENTER,
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              spacing: { after: 0 },
              children: [run(header, { size: 11, bold: true })],
            }),
          ],
        }),
    ),
  });

  const dataRows = rows.map(
    (row) =>
      new TableRow({
        children: row.map(
          (cell, index) =>
            new TableCell({
              borders: CELL_BORDERS,
              width: { size: widths[index], type: WidthType.DXA },
              verticalAlign: VerticalAlign.CENTER,
              children: [
                new Paragraph({
                  alignment:
                    alignments[index] === "left"
                      ? AlignmentType.LEFT
                      : AlignmentType.CENTER,
                  spacing: { after: 0 },
                  children: [run(String(cell ?? ""), { size: 11 })],
                }),
              ],
            }),
        ),
      }),
  );

  return new Table({
    columnWidths: widths,
    layout: TableLayoutType.FIXED,
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    rows: [headerRow, ...dataRows],
  });
}

function pngSize(buffer) {
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

function buildFigure(block, figureDir, number) {
  const absolute = path.join(figureDir, block.file);
  if (!fs.existsSync(absolute)) {
    throw new Error(
      `Thiếu hình: ${absolute}. Chạy: python scripts/report_figures.py`,
    );
  }
  const data = fs.readFileSync(absolute);
  const size = pngSize(data);
  const widthPx = Math.round(((block.widthCm ?? 15) / 2.54) * 96);
  const heightPx = Math.round((size.height / size.width) * widthPx);
  const image = new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [
      new ImageRun({
        type: "png",
        data,
        transformation: { width: widthPx, height: heightPx },
        altText: {
          title: `Hình ${number}`,
          description: block.caption,
          name: block.file,
        },
      }),
    ],
  });
  return [image, captionParagraph(`Hình ${number}. ${block.caption}`)];
}

function buildCode(block) {
  const lines = block.text.replace(/\r\n/g, "\n").split("\n");
  return lines.map(
    (line) =>
      new Paragraph({
        style: "CodeBlock",
        children: [run(line, { size: 10, mono: true })],
      }),
  );
}

function coverChildren(meta) {
  const children = [];
  for (let index = 0; index < 4; index += 1) {
    children.push(new Paragraph({ children: [] }));
  }
  children.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [run(meta.title, { size: 20, bold: true })],
    }),
  );
  children.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 360 },
      children: [run(meta.subject, { size: 14 })],
    }),
  );
  children.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 720 },
      children: [run(meta.subtitle, { size: 13, italics: true })],
    }),
  );
  for (let index = 0; index < 3; index += 1) {
    children.push(new Paragraph({ children: [] }));
  }
  for (const [label, value] of meta.fields) {
    children.push(
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 120 },
        children: [
          run(`${label}: `, { size: 13, bold: true }),
          run(value, { size: 13 }),
        ],
      }),
    );
  }
  for (let index = 0; index < 5; index += 1) {
    children.push(new Paragraph({ children: [] }));
  }
  children.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [run(meta.place, { size: 13, italics: true })],
    }),
  );
  return children;
}

function documentStyles() {
  const headingStyle = (id, name, size, outlineLevel, extra = {}) => ({
    id,
    name,
    basedOn: "Normal",
    next: "Normal",
    quickFormat: true,
    run: { font: FONT, size: size * 2, bold: true, color: "000000", ...extra },
    paragraph: {
      spacing: { before: 240, after: 120 },
      alignment: AlignmentType.LEFT,
      keepNext: true,
      outlineLevel,
    },
  });

  return {
    default: {
      document: {
        run: { font: FONT, size: 26 },
        paragraph: {
          spacing: { line: 360, lineRule: LineRuleType.AUTO, after: 120 },
          alignment: AlignmentType.JUSTIFIED,
        },
      },
    },
    paragraphStyles: [
      headingStyle("Heading1", "Heading 1", 16, 0),
      headingStyle("Heading2", "Heading 2", 14, 1),
      headingStyle("Heading3", "Heading 3", 13, 2, { italics: true }),
      {
        id: "CodeBlock",
        name: "Code Block",
        basedOn: "Normal",
        run: { font: MONO, size: 20 },
        paragraph: {
          spacing: { line: 240, lineRule: LineRuleType.AUTO, after: 0 },
          alignment: AlignmentType.LEFT,
          indent: { left: 283 },
          shading: { fill: "F2F2F2", type: ShadingType.CLEAR },
        },
      },
    ],
  };
}

export async function writeReport({ blocks, figuresDir, outPath, meta }) {
  const figureDir =
    typeof figuresDir === "string" ? figuresDir : fileURLToPath(figuresDir);
  const figures = [];
  const tables = [];
  const numberingConfig = [
    {
      reference: "report-bullets",
      levels: [
        {
          level: 0,
          format: LevelFormat.BULLET,
          text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        },
      ],
    },
  ];

  blocks.forEach((block, index) => {
    if (block.type === "figure") figures.push(block.caption);
    if (block.type === "table") tables.push(block.caption);
    if (block.type === "numbered") {
      numberingConfig.push({
        reference: `report-numbered-${index}`,
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      });
    }
  });

  const children = [...coverChildren(meta), pageBreak(), sectionTitle("MỤC LỤC")];
  children.push(
    new TableOfContents("Mục lục", {
      hyperlink: true,
      headingStyleRange: "1-3",
    }),
  );
  children.push(pageBreak(), sectionTitle("DANH MỤC HÌNH"));
  if (figures.length === 0) {
    children.push(bodyParagraph("(chưa có hình)"));
  }
  figures.forEach((caption, index) => {
    children.push(bodyParagraph(`Hình ${index + 1}. ${caption}`));
  });
  children.push(pageBreak(), sectionTitle("DANH MỤC BẢNG"));
  if (tables.length === 0) {
    children.push(bodyParagraph("(chưa có bảng)"));
  }
  tables.forEach((caption, index) => {
    children.push(bodyParagraph(`Bảng ${index + 1}. ${caption}`));
  });
  children.push(pageBreak());

  let figureNumber = 0;
  let tableNumber = 0;
  blocks.forEach((block, index) => {
    switch (block.type) {
      case "h1":
        children.push(heading(1, block.text));
        break;
      case "h2":
        children.push(heading(2, block.text));
        break;
      case "h3":
        children.push(heading(3, block.text));
        break;
      case "p":
        children.push(bodyParagraph(block.text));
        break;
      case "bullets":
        for (const item of block.items) {
          children.push(
            new Paragraph({
              numbering: { reference: "report-bullets", level: 0 },
              children: [run(item)],
            }),
          );
        }
        break;
      case "numbered":
        for (const item of block.items) {
          children.push(
            new Paragraph({
              numbering: { reference: `report-numbered-${index}`, level: 0 },
              children: [run(item)],
            }),
          );
        }
        break;
      case "table":
        tableNumber += 1;
        children.push(captionParagraph(`Bảng ${tableNumber}. ${block.caption}`));
        children.push(buildTable(block));
        children.push(new Paragraph({ children: [] }));
        break;
      case "figure": {
        figureNumber += 1;
        children.push(...buildFigure(block, figureDir, figureNumber));
        break;
      }
      case "code":
        children.push(...buildCode(block));
        children.push(new Paragraph({ children: [] }));
        break;
      case "pagebreak":
        children.push(pageBreak());
        break;
      default:
        throw new Error(`Unknown block type: ${block.type}`);
    }
  });

  const document = new Document({
    styles: documentStyles(),
    numbering: { config: numberingConfig },
    sections: [
      {
        properties: {
          page: {
            size: { width: 11906, height: 16838 },
            margin: { top: 1134, bottom: 1134, left: 1701, right: 1134 },
          },
        },
        headers: {
          default: new Header({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  run("Báo cáo đồ án — Student Social Media & Wellbeing", {
                    size: 9,
                    color: "808080",
                  }),
                ],
              }),
            ],
          }),
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  run("Trang ", { size: 9 }),
                  new TextRun({ children: [PageNumber.CURRENT], size: 18 }),
                  run(" / ", { size: 9 }),
                  new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18 }),
                ],
              }),
            ],
          }),
        },
        children,
      },
    ],
  });

  const buffer = await Packer.toBuffer(document);
  fs.writeFileSync(outPath, buffer);
}
