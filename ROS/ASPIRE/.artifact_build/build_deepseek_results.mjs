import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const sourceFile = "/private/tmp/deepseek_fix_loop_progress.md";
const outputDir = "/Users/ronald/Developer/ROS/ASPIRE/outputs/deepseek-results-20260919";
const outputFile = `${outputDir}/deepseek_per_task_results.xlsx`;
const previewSummary = `${outputDir}/preview_summary.png`;
const previewDetail = `${outputDir}/preview_detail.png`;
const progressPath = "/home/user/ASPIRE/aspire/sim/docs/progress/fix_loop_progress.md";
const font = "Arial";

const md = await fs.readFile(sourceFile, "utf8");
const parsed = [];
let suite = null;
for (const line of md.split("\n")) {
  const heading = line.match(/^## (libero_[a-z_]+)\s+\(/);
  if (heading) {
    suite = heading[1];
    continue;
  }
  if (!suite || !line.startsWith("| ") || line.startsWith("| Task ") || line.startsWith("|---")) continue;
  const [taskDisplay, status, rate] = line.slice(1, -1).split("|").map((v) => v.trim());
  if (taskDisplay === "debug_smoke") continue;
  const match = rate.match(/(\d+)\/(\d+) \((\d+)%\) \[run ([0-9a-f]+)\]/);
  parsed.push({
    suite,
    taskDisplay,
    status,
    passes: match ? Number(match[1]) : null,
    trials: match ? Number(match[2]) : null,
    passRate: match ? Number(match[3]) / 100 : null,
    runId: match ? match[4] : "",
  });
}

function taskId(suite, display) {
  if (suite.startsWith("libero_goal_")) {
    const map = {
      "open middle_drawer_of_the_cabinet": "open_the_middle_drawer_of_the_cabinet",
      "open top_drawer + bowl": "open_the_top_drawer_and_put_the_bowl_inside",
      "push plate → stove front": "push_the_plate_to_the_front_of_the_stove",
      "bowl → plate": "put_the_bowl_on_the_plate",
      "bowl → stove": "put_the_bowl_on_the_stove",
      "bowl → top of cabinet": "put_the_bowl_on_top_of_the_cabinet",
      "cream_cheese → bowl": "put_the_cream_cheese_in_the_bowl",
      "wine_bottle → rack": "put_the_wine_bottle_on_the_rack",
      "wine_bottle → top of cabinet": "put_the_wine_bottle_on_top_of_the_cabinet",
      "turn → stove": "turn_on_the_stove",
    };
    return map[display];
  }
  if (suite.startsWith("libero_object_")) {
    const item = display.split(" → ")[0];
    return `pick_up_the_${item}_and_place_it_in_the_basket`;
  }
  const location = display.replace("bowl: ", "").split(" → ")[0];
  return `pick_up_the_black_bowl_${location}_and_place_it_on_the_plate`;
}

function isManifestBacked(row) {
  if (!row.runId) return false;
  if (row.suite === "libero_goal_swap" || row.suite === "libero_object_swap") return true;
  if (row.suite === "libero_spatial_swap") return true;
  if (row.suite === "libero_spatial_task") {
    return new Set([
      "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
      "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
      "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
      "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
      "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    ]).has(row.taskId);
  }
  return false;
}

const rows = parsed.map((row) => {
  const id = taskId(row.suite, row.taskDisplay);
  const enriched = { ...row, taskId: id };
  const manifest = isManifestBacked(enriched);
  return {
    ...enriched,
    evidence: manifest ? "Manifest" : "Generated progress",
    sourcePath: manifest
      ? `/home/user/ASPIRE/aspire/sim/outputs/libero_fix_loop_eval/${row.suite}/${id}/runs/${row.runId}/manifest.json`
      : progressPath,
  };
});

if (rows.length !== 60) throw new Error(`Expected 60 canonical task rows, got ${rows.length}`);

const suites = [...new Set(rows.map((r) => r.suite))].sort();
const workbook = Workbook.create();
const summary = workbook.worksheets.add("结果汇总");
const detail = workbook.worksheets.add("逐任务明细");
summary.showGridLines = false;
detail.showGridLines = false;
summary.tabColor = "#1F4E78";
detail.tabColor = "#5B9BD5";

summary.getRange("A2:F2").merge();
summary.getRange("A2").values = [["DeepSeek LIBERO-Pro 逐任务结果汇总"]];
summary.getRange("A2").format = { font: { name: font, size: 16, bold: true, color: "#17365D" } };
summary.getRange("A3:F3").merge();
summary.getRange("A3").values = [["模型：deepseek-flash（由 Claude 配置与启动记录推断）｜Stage 2：held-out seeds 1–50｜截至 2026-09-19"]];
summary.getRange("A3").format = { font: { name: font, size: 10, italic: true, color: "#595959" } };
summary.getRange("A4:F4").format.borders = { bottom: { style: "thin", color: "#9EADBA" } };

summary.getRange("A6:G6").values = [["Suite", "标准任务数", "已完成", "通过数", "Trials", "通过率", "仅 Stage 1"]];
summary.getRange("A7:A12").values = suites.map((s) => [s]);
for (let r = 7; r <= 12; r++) {
  summary.getRange(`B${r}`).formulas = [[`=COUNTIFS('逐任务明细'!$A$7:$A$66,A${r})`]];
  summary.getRange(`C${r}`).formulas = [[`=COUNTIFS('逐任务明细'!$A$7:$A$66,A${r},'逐任务明细'!$D$7:$D$66,"done")`]];
  summary.getRange(`D${r}`).formulas = [[`=SUMIFS('逐任务明细'!$E$7:$E$66,'逐任务明细'!$A$7:$A$66,A${r})`]];
  summary.getRange(`E${r}`).formulas = [[`=SUMIFS('逐任务明细'!$F$7:$F$66,'逐任务明细'!$A$7:$A$66,A${r})`]];
  summary.getRange(`F${r}`).formulas = [[`=IF(E${r}=0,"",D${r}/E${r})`]];
  summary.getRange(`G${r}`).formulas = [[`=COUNTIFS('逐任务明细'!$A$7:$A$66,A${r},'逐任务明细'!$D$7:$D$66,"stage1-done")`]];
}
summary.getRange("A13").values = [["总计"]];
summary.getRange("B13").formulas = [["=SUM(B7:B12)"]];
summary.getRange("C13").formulas = [["=SUM(C7:C12)"]];
summary.getRange("D13").formulas = [["=SUM(D7:D12)"]];
summary.getRange("E13").formulas = [["=SUM(E7:E12)"]];
summary.getRange("F13").formulas = [["=IF(E13=0,\"\",D13/E13)"]];
summary.getRange("G13").formulas = [["=SUM(G7:G12)"]];
summary.getRange("A6:G6").format = {
  fill: "#1F4E78",
  font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { insideVertical: { style: "thin", color: "#FFFFFF" }, bottom: { style: "thin", color: "#17365D" } },
};
summary.getRange("A7:G12").format = { font: { name: font, size: 10, color: "#222222" }, verticalAlignment: "center", borders: { bottom: { style: "thin", color: "#D9E2F3" } } };
summary.getRange("A13:G13").format = { fill: "#D9EAF7", font: { name: font, size: 10, bold: true, color: "#17365D" }, borders: { top: { style: "medium", color: "#1F4E78" }, bottom: { style: "double", color: "#1F4E78" } } };
summary.getRange("B7:E13").format.numberFormat = "#,##0";
summary.getRange("F7:F13").format.numberFormat = "0.0%";
summary.getRange("G7:G13").format.numberFormat = "#,##0";
summary.getRange("B7:G13").format.horizontalAlignment = "right";
summary.getRange("A15:G17").values = [
  ["口径说明", null, null, null, null, null, null],
  ["模型归属", "DeepSeek 名称未写入 trial manifest；根据 2026-09-14 05:21 后生效的 Claude 配置（所有模型别名均指向 deepseek-flash）及启动历史推断。", null, null, null, null, null],
  ["证据级别", "33 个任务有本机 manifest；其余 27 个任务由服务器自动生成的 fix_loop_progress.md 恢复，其中含 2 个仅完成 Stage 1 的任务。", null, null, null, null, null],
];
summary.getRange("A15:G15").format = { fill: "#D9E2F3", font: { name: font, size: 10, bold: true, color: "#17365D" } };
summary.getRange("A16:A17").format.font = { name: font, size: 10, bold: true, color: "#404040" };
summary.getRange("B16:G17").merge(true);
summary.getRange("B16:G17").format = { font: { name: font, size: 10, color: "#404040" }, wrapText: true, verticalAlignment: "top" };
summary.getRange("A16:G17").format.rowHeight = 34;
summary.getRange("A2:G17").format.verticalAlignment = "center";
summary.getRange("A:A").format.columnWidth = 28;
summary.getRange("B:E").format.columnWidth = 13;
summary.getRange("F:G").format.columnWidth = 14;

const chart = summary.charts.add("bar", [summary.getRange("A6:A12"), summary.getRange("F6:F12")]);
chart.title = "各 Suite 通过率";
chart.titleTextStyle.typeface = font;
chart.titleTextStyle.fontSize = 14;
chart.hasLegend = false;
chart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 10 } };
chart.yAxis = { numberFormatCode: "0%", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 10 } };
chart.setPosition("I5", "P18");

detail.getRange("A2:L2").merge();
detail.getRange("A2").values = [["DeepSeek 逐任务明细"]];
detail.getRange("A2").format = { font: { name: font, size: 16, bold: true, color: "#17365D" } };
detail.getRange("A3:L3").merge();
detail.getRange("A3").values = [[`Source: user@10.36.116.25:${progressPath}`]];
detail.getRange("A3").format = { font: { name: font, size: 9, italic: true, color: "#666666" } };
detail.getRange("A4:L4").merge();
detail.getRange("A4").values = [["注意：trial 路径中的 aws_anthropic_bedrock-claude-sonnet-4-6 是模拟器策略目录名，不是生成修复代码的推理模型。"]];
detail.getRange("A4").format = { fill: "#FFF2CC", font: { name: font, size: 9, color: "#7F6000" } };
detail.getRange("A6:L6").values = [["Suite", "Task ID", "任务简称", "状态", "通过数", "Trials", "通过率", "Run ID", "证据", "模型", "模型归属", "服务器源路径"]];
detail.getRange("A7:L66").values = rows.map((r) => [
  r.suite, r.taskId, r.taskDisplay, r.status, r.passes, r.trials, r.passRate, r.runId,
  r.evidence, "deepseek-flash", "配置与启动记录推断", r.sourcePath,
]);
detail.getRange("A6:L6").format = {
  fill: "#1F4E78",
  font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { insideVertical: { style: "thin", color: "#FFFFFF" }, bottom: { style: "thin", color: "#17365D" } },
};
detail.getRange("A7:L66").format = { font: { name: font, size: 9, color: "#222222" }, verticalAlignment: "center", borders: { bottom: { style: "thin", color: "#E7E6E6" } } };
detail.getRange("E7:F66").format.numberFormat = "#,##0";
detail.getRange("G7:G66").format.numberFormat = "0.0%";
detail.getRange("E7:G66").format.horizontalAlignment = "right";
detail.getRange("D7:D66").conditionalFormats.add("containsText", { text: "stage1-done", format: { fill: "#FFF2CC", font: { color: "#9C6500", bold: true } } });
detail.getRange("G7:G66").conditionalFormats.add("colorScale", { colors: ["#F8696B", "#FFEB84", "#63BE7B"], thresholds: ["min", { type: "percentile", value: 50 }, "max"] });
detail.tables.add("A6:L66", true, "DeepSeekTaskResults").style = "TableStyleMedium2";
detail.freezePanes.freezeRows(6);
detail.freezePanes.freezeColumns(2);
detail.getRange("A:A").format.columnWidth = 22;
detail.getRange("B:B").format.columnWidth = 52;
detail.getRange("C:C").format.columnWidth = 32;
detail.getRange("D:D").format.columnWidth = 14;
detail.getRange("E:G").format.columnWidth = 11;
detail.getRange("H:H").format.columnWidth = 19;
detail.getRange("I:I").format.columnWidth = 20;
detail.getRange("J:J").format.columnWidth = 18;
detail.getRange("K:K").format.columnWidth = 24;
detail.getRange("L:L").format.columnWidth = 86;
detail.getRange("B7:C66").format.wrapText = false;
detail.getRange("L7:L66").format.wrapText = false;

workbook.recalculate();
const summaryInspect = await workbook.inspect({ kind: "table", range: "结果汇总!A2:G17", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 10 });
console.log(summaryInspect.ndjson);
const detailInspect = await workbook.inspect({ kind: "table", range: "逐任务明细!A6:L15", include: "values,formulas", tableMaxRows: 12, tableMaxCols: 12 });
console.log(detailInspect.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const summaryPng = await workbook.render({ sheetName: "结果汇总", range: "A1:P19", scale: 1.4, format: "png" });
await fs.writeFile(previewSummary, new Uint8Array(await summaryPng.arrayBuffer()));
const detailPng = await workbook.render({ sheetName: "逐任务明细", range: "A1:L20", scale: 1.1, format: "png" });
await fs.writeFile(previewDetail, new Uint8Array(await detailPng.arrayBuffer()));
const out = await SpreadsheetFile.exportXlsx(workbook);
await out.save(outputFile);
console.log(JSON.stringify({ outputFile, previewSummary, previewDetail, rowCount: rows.length }));
