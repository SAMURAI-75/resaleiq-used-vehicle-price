// IEEE conference-format paper (US Letter, two columns, Times New Roman 10 pt) -> docs/paper/ResaleIQ_IEEE_Paper.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, SectionType, TabStopType, VerticalAlign, ShadingType,
} = require("docx");

const ROOT = "/Users/fouzan/Desktop/AiML Project";
const FIG = path.join(ROOT, "docs/paper/figures");
const OUT = path.join(ROOT, "docs/paper/ResaleIQ_IEEE_Paper.docx");
const FONT = "Times New Roman";
const COL = 5040;                 // one column, DXA (3.5 in)

// ---- inline markup: *italic*, **bold**, _{sub}, ^{sup}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), font: FONT, bold: true, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), font: FONT, italics: true, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), font: FONT, subScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), font: FONT, superScript: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, o = {}) => new Paragraph({
  children: runs(text, { size: o.size || 20 }),
  alignment: o.align || AlignmentType.JUSTIFIED,
  indent: o.noIndent ? undefined : { firstLine: 202 },
  spacing: { after: o.after ?? 0, before: o.before ?? 0, line: 228 },
  keepNext: o.keepNext,
});
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"];
let h1n = 0, h2n = 0;
const H1 = (t) => { h2n = 0; return new Paragraph({
  children: [new TextRun({ text: `${ROMAN[h1n++]}. `, font: FONT, size: 20 }), new TextRun({ text: t, font: FONT, size: 20, smallCaps: true })],
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true, keepLines: true }); };
const H2 = (t) => new Paragraph({
  children: [new TextRun({ text: `${String.fromCharCode(65 + h2n++)}. ${t}`, font: FONT, size: 20, italics: true })],
  spacing: { before: 120, after: 60 }, keepNext: true, keepLines: true });

// ---- tables (8 pt, caption above in IEEE style)
let tn = 0;
function TABLE(title, header, rows, widths, opts = {}) {
  const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const cell = (txt, w, isHead, rowIdx, colIdx) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: "F2F2F2" } : undefined,
    borders: { top: isHead ? thin : none, bottom: (isHead || rowIdx === rows.length - 1) ? thin : none, left: none, right: none },
    children: [new Paragraph({
      children: runs(String(txt), { size: 15, bold: isHead || (opts.boldRows || []).includes(rowIdx) }),
      alignment: colIdx === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { after: 0, line: 200 } })],
  });
  const roman = ROMAN[tn++];
  return [
    new Paragraph({ children: [new TextRun({ text: `TABLE ${roman}`, font: FONT, size: 16 })], alignment: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: title, font: FONT, size: 16, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { after: 80 }, keepNext: true }),
    new Table({
      width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
      alignment: AlignmentType.CENTER,
      rows: [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, j) => cell(h, widths[j], true, -1, j)) }),
             ...rows.map((r, i) => new TableRow({ cantSplit: true, children: r.map((c, j) => cell(c, widths[j], false, i, j)) }))],
    }),
    new Paragraph({ children: [], spacing: { after: 80 } }),
  ];
}

// ---- figures (column width, caption below)
let fn = 0;
function FIGURE(file, w, h, caption) {
  const px = 322;                                  // 3.35 in at 96 dpi
  return [
    new Paragraph({ children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)),
      transformation: { width: px, height: Math.round(px * h / w) } })], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true }),
    new Paragraph({ children: runs(`Fig. ${++fn}. ${caption}`, { size: 16 }), alignment: AlignmentType.JUSTIFIED, spacing: { after: 140, line: 200 } }),
  ];
}

const EQ = (text, n) => new Paragraph({
  children: [new TextRun({ text: "\t", font: FONT }), ...runs(text, { size: 18 }), new TextRun({ text: `\t(${n})`, font: FONT, size: 20 })],
  tabStops: [{ type: TabStopType.CENTER, position: COL / 2 }, { type: TabStopType.RIGHT, position: COL }],
  spacing: { before: 80, after: 80 } });

// ================================================================ content
const authors = [
  ["Kartik Katwe", "1MS24CS079"], ["Mohammed Fouzan Anwar", "1MS24CS101"],
  ["Sathvi Nagraj", "1MS24CS233"], ["Ishaan Nandi Roy", "1MS24CS069"],
];
const authorTable = new Table({
  width: { size: 10440, type: WidthType.DXA }, columnWidths: [2610, 2610, 2610, 2610], alignment: AlignmentType.CENTER,
  borders: { top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
             left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
             insideHorizontal: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
  rows: [new TableRow({ children: authors.map(([n, usn]) => new TableCell({
    width: { size: 2610, type: WidthType.DXA },
    children: [
      new Paragraph({ children: [new TextRun({ text: n, font: FONT, size: 22 })], alignment: AlignmentType.CENTER, spacing: { after: 20 } }),
      ...[`Dept. of Computer Science and Engineering`, `Ramaiah Institute of Technology`, `Bengaluru, India`, `${usn.toLowerCase()}@msrit.edu`].map((t, i) =>
        new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 18, italics: i === 0 })], alignment: AlignmentType.CENTER, spacing: { after: 0 } })),
    ] })) })],
});

const front = [
  new Paragraph({ children: [new TextRun({ text: "Used Two-Wheeler Resale Price Prediction in India with Electric-Vehicle Depreciation Analysis", font: FONT, size: 48 })],
    alignment: AlignmentType.CENTER, spacing: { after: 240 } }),
  authorTable,
  new Paragraph({ children: [], spacing: { after: 200 } }),
];

const body = [
  new Paragraph({ children: [new TextRun({ text: "Abstract", font: FONT, size: 18, bold: true, italics: true }),
    new TextRun({ text: "—India is the world's largest two-wheeler market and electric scooters are now being resold in large numbers, yet there is no independent way to price a used one, and most machine-learning resale studies cover cars. We replicate the four algorithms of a recent IEEE two-wheeler pricing paper (Decision Tree, SVR, KNN, SGD) on 7,066 cleaned Indian listings and compare them with Linear Regression, Random Forest and XGBoost on the same split; XGBoost reaches R² 0.921 against 0.864 for the base paper's best model. A model trained on 2020–21 listings under-prices 13,569 live 2026 listings by 8%, so we collect 30,306 two-wheeler and 91,748 car advertisements from November 2026 and train monotone, seed-averaged XGBoost models with Tree-SHAP explanations. The two-wheeler model's typical error is 12.4% (noise floor 11.1%); the car model's 8.2% beats the listing site's own valuation (11.3%) on the same ads. With variant and city held fixed, used electric two-wheelers lose 12.0% of their value per year against 6.3% for petrol ones (p = 0.001). We also build a year-by-year record of India's national tax on new vehicles from 1990 to 2026 and show that adding it to the models changes accuracy by under 0.1 point.", font: FONT, size: 18, bold: true })],
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 216 } }),
  new Paragraph({ children: [new TextRun({ text: "Index Terms", font: FONT, size: 18, bold: true, italics: true }),
    new TextRun({ text: "—resale price prediction, two-wheelers, electric vehicles, depreciation, XGBoost, SHAP, India", font: FONT, size: 18, bold: true })],
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 } }),

  H1("Introduction"),
  P("India sells more two-wheelers than any other country, and its used market is larger still. Electric two-wheelers moved from niche to mainstream between 2021 and 2026, so the first large wave of them is now being resold. Buyers, sellers, dealers and lenders need a fair reference price, but used prices are set largely by guesswork or by listing-site estimates whose methods are not published. For electric scooters there is no resale index and no standard battery-health certificate in India."),
  P("Machine-learning (ML) pricing research has focused on cars: 13 of the 19 studies we surveyed price cars, and the four two-wheeler studies use only petrol bikes, never test their models on newer data and, in two cases, report classification \"accuracy\" for a regression task [1], [4]. Studies of electric-vehicle (EV) resale value exist only for cars, mostly in the United States [18], [19]."),
  P("We ask three questions. *RQ1:* do tree ensembles beat the base paper's algorithms when all are scored with regression metrics on the same split? *RQ2:* does a model trained on 2020–21 listings still price 2026 listings correctly? *RQ3:* do used electric two-wheelers lose value faster than petrol ones of the same variant? Our contributions are (i) the first same-variant estimate of EV versus petrol depreciation for Indian two-wheelers; (ii) a temporal-validity test showing that an older price model goes stale; (iii) a fair re-implementation of the base paper with proper metrics; (iv) deployed, explained 2026 price models benchmarked against a commercial valuation; and (v) a documented 1990–2026 record of Indian vehicle taxes, tested as a model input."),

  H1("Related Work"),
  P("Ferdous *et al.* [1] predict used motorcycle prices in Bangladesh with Decision Tree, SVR, KNN and SGD and report 97% \"accuracy\" for the Decision Tree; this is our base paper. Indian two-wheeler studies use tree models with SHAP or LIME [2], compare linear, SVR, Random Forest and XGBoost models on about 5,000 Kaggle listings [3], or use a hybrid Naive Bayes–KNN classifier [4]. None includes electric vehicles."),
  P("For cars, Random Forest or gradient boosting is the best model in most studies [5], [9], [11]–[14]; XGBoost with SHAP is a common toolkit [5], [6]; and images can add signal [15]. Bukvić *et al.* [8] are the only authors who re-test a model on later data and find that its accuracy falls. Roberson *et al.* [18] show that battery-electric cars in the US keep less of their value than petrol cars, and a Michigan thesis finds the same on about 150,000 listings [19]. Our work applies these ideas to Indian two-wheelers and adds a temporal test, a same-variant EV design and a commercial benchmark."),

  H1("Data"),
  P("Table I lists the data. The Kaggle set [22] repeats most rows: 25,324 of 32,648 are exact duplicates. The 2026 advertisements were collected from BikeWale and CarWale in November 2026 [24], [25], city by city, at one request every 0.4 s. No seller names, phone numbers or addresses were stored. Each 2026 ad has its make, model, exact variant, model year, asking price, kilometres, owners, fuel, body type, city and seller type. Car ads also carry the variant's new on-road price and CarWale's own valuation, which we use only as a benchmark, never as a feature."),
  ...TABLE("Datasets", ["Dataset", "Source, period", "Rows used"], [
    ["Used bikes (Kaggle) [22]", "Listings 2020–21", "7,066"],
    ["BikeWale used ads [24]", "12–13 Nov 2026", "28,062 (891 EV)"],
    ["CarWale used ads [25]", "13 Nov 2026", "88,800 (1,632 EV)"],
    ["Official catalogue", "Brand and model pages", "541 2W, 345 car models"],
    ["Tax history (ours)", "Budgets, GST Council, 1990–2026", "43,978 variant-years"],
  ], [1900, 1700, 1440]),
  P("Cleaning removes duplicates, re-posted ads, listings with implausible kilometres, prices above 1.6× or below 3% of the new price, and outliers beyond 1.5×IQR within each model × age group of ten or more ads. Free-text names are mapped to official model names (98.4% of Kaggle rows), and cities are mapped to states with a 1,578-city list (99.8% of bike ads). Hybrids are separated from EVs. Each model uses an 80/20 random split with seed 42."),

  H1("Methodology"),
  H2("Features and Model"),
  P("Categorical inputs are brand, model, variant, fuel, body type, gearbox, state, city and seller type; numeric inputs are age, log(1 + km), owners and the log new price of the exact variant and of the model. The target is log(price), because prices range from ₹5,000 to over ₹1 crore. The proposed model is XGBoost [20] with native categorical splits and monotone constraints: the price may never rise with age, kilometres or owners, and never fall as the new price rises. Five boosted models with different seeds (three for cars) are averaged. Settings are learning rate 0.03, depth 10 (two-wheelers) or 8 (cars), minimum child weight 5, row and column subsampling 0.8 and 0.7, L2 penalty 3, and early stopping after 200 rounds on a 15% validation part of the training data, followed by a refit on the full training split."),
  H2("Explanations and Price Range"),
  P("Tree SHAP [21], computed exactly through XGBoost, splits every prediction into additive per-feature contributions, averaged across the seed models. Each price also has a range: the middle 60% of validation errors for that vehicle segment."),
  H2("EV Depreciation Model"),
  P("On 2026 ads up to seven years old, the log price is regressed on age, log(1 + km) and an EV × age interaction, with a fixed effect for each BikeWale variant and each city with 30 or more ads; standard errors are clustered by variant:", { keepNext: true }),
  EQ("log *P* = β_{1}·age + β_{2}·log(1+km) + β_{3}·(EV×age) + α_{v} + γ_{c} + ε", 1),
  P("The variant effect absorbs brand, engine, fuel and new price, so the age slope compares the same variant at different ages. The yearly loss is exp(β_{1}) − 1 for petrol and exp(β_{1} + β_{3}) − 1 for EVs; the hypothesis is tested on β_{3}.", { noIndent: true }),
  H2("Indian Tax History, 1990–2026"),
  P("A used vehicle's original price includes the tax paid when it was new, and India's rates changed many times. We record, for every year from 1990, the effective national tax on the price before tax: central excise plus state VAT or sales tax until June 2017, GST plus compensation cess from July 2017, and the reformed GST from 22 September 2025, together with EV rates and state road-tax slabs for 11 states. Each rule carries a source and a confidence level [26]–[28]; rates before 2008 are poorly documented and marked low confidence. Cars are classed small, mid-size, large or SUV from the engine size in the variant name and the body type. Table II shows the effective rates; Fig. 5 plots them. Two inputs are derived for each ad: the rate paid in its model year, and log(1 + rate then) − log(1 + rate today)."),
  ...TABLE("Effective national tax on a new vehicle (%)", ["Year", "2W 150cc", "2W 450cc", "E-2W", "Small car", "Large SUV"], [
    ["1995*", "36.4", "36.4", "16.6", "76.0", "76.0"],
    ["2005", "30.5", "30.5", "19.2", "39.5", "39.5"],
    ["2010", "23.8", "23.8", "19.2", "23.8", "37.2"],
    ["2014", "21.5", "21.5", "19.2", "21.5", "39.5"],
    ["2016", "28.2", "28.2", "21.4", "30.0", "53.4"],
    ["2018", "28.0", "31.0", "12.0", "29.0", "50.0"],
    ["2020", "28.0", "31.0", "5.0", "29.0", "50.0"],
    ["2026", "18.0", "40.0", "5.0", "18.0", "40.0"],
  ], [740, 860, 860, 780, 900, 900]),
  P("*Low confidence. Excise + VAT before July 2017, GST + cess after; 1 July of each year.", { size: 15, noIndent: true, after: 80 }),

  H1("Experiments and Results"),
  H2("Base-Paper Replication (RQ1)"),
  P("All seven models are trained on the same 80% of the Kaggle data and scored on the remaining 1,414 bikes (Table III, Fig. 1). XGBoost has the lowest error on every metric. Tree ensembles win because a bike's value depends on interactions — a six-year-old Royal Enfield holds its value, a six-year-old commuter bike does not — that one deep tree over-fits and a linear model cannot express. SGD's negative R² comes from a few very large misses on superbikes. The base paper's 97% \"accuracy\" cannot be reproduced as a regression metric: its best model prices 64.8% of bikes within ±20%."),
  ...TABLE("Kaggle test split (1,414 bikes)", ["Method", "MAE (₹)", "R²", "Within ±20%"], [
    ["XGBoost (proposed)", "10,422", "0.921", "79.8%"],
    ["Random Forest", "11,399", "0.910", "76.0%"],
    ["SVR [1]", "12,355", "0.816", "76.5%"],
    ["Linear Regression", "13,209", "0.851", "73.0%"],
    ["KNN [1]", "13,780", "0.883", "70.2%"],
    ["Decision Tree [1]", "14,332", "0.864", "64.8%"],
    ["SGD [1]", "21,906", "−1.29", "70.2%"],
  ], [1900, 1040, 900, 1200], { boldRows: [0] }),
  ...FIGURE("m4_model_comparison.png", 1650, 570, "Model comparison on the Kaggle test split: R² and share of bikes priced within ±20%."),
  H2("Temporal Validity (RQ2)"),
  P("The Kaggle-trained XGBoost is scored on 13,569 live 2026 BikeWale ads of models it knows. Table IV gives the ratio of real to predicted price by age. The model under-prices 2026 ads by 8% overall and by about 16% for bikes 5–12 years old, while new bikes are about right: new-vehicle prices rose faster than inflation, and older bikes inherited the rise. Age-band calibration fixes the level, but R² (log) on 2026 ads is only 0.67 against 0.92 in-sample, so the main models were retrained on 2026 data."),
  ...TABLE("Real ÷ predicted price on 2026 ads, by age (years)", ["Age", "≤1", "1–3", "3–5", "5–8", "8–12", "12–20", "All"], [
    ["Factor", "0.98", "1.01", "1.07", "1.16", "1.16", "1.14", "1.08"],
  ], [720, 600, 620, 620, 620, 620, 620, 620]),
  H2("Main 2026 Models"),
  P("Table V scores the 2026 models on 20% held-out ads. The noise floor — the median gap between asking prices of the very same variant, year and seller type — is 11.1% for two-wheelers and 8.7% for cars; no model using these inputs can do much better. The two-wheeler model's typical miss of 12.4% and the car model's 8.4% are close to these floors. On the 16,650 test cars that show CarWale's valuation, our model's typical miss is 8.2% against 11.3% for CarWale's own estimate. Electric scooters are the hardest segment (14.8%), consistent with buyers' uncertainty about battery health. Other losses, learning rates and regularisation all changed results by under 0.3 point; seed averaging added 0.8 point within ±20%."),
  ...TABLE("2026 models, held-out ads (typical miss = median |% error|)", ["Segment", "Ads", "Typical miss", "±20%", "R² (log)"], [
    ["Two-wheelers, all", "5,613", "12.4%", "69.4%", "0.869"],
    ["  Petrol motorcycles", "4,532", "12.0%", "69.8%", "0.869"],
    ["  Petrol scooters", "912", "13.7%", "67.9%", "0.782"],
    ["  Electric scooters", "138", "14.8%", "62.3%", "0.656"],
    ["  Simple rule", "4,014", "14.0%", "65.9%", "0.862"],
    ["Cars, all", "17,760", "8.4%", "83.1%", "0.966"],
    ["  Electric cars", "342", "7.2%", "89.5%", "0.957"],
    ["  Ours, valued ads", "16,650", "8.2%", "83.9%", "0.969"],
    ["  CarWale valuation", "16,650", "11.3%", "73.2%", "0.949"],
  ], [1620, 780, 1000, 820, 820], { boldRows: [0, 5] }),
  ...FIGURE("m9_market_models.png", 2006, 631, "Typical miss of the 2026 models by segment, with the noise floor (dashed)."),
  ...FIGURE("m8_shap_importance.png", 900, 510, "Mean |SHAP| by input on the Kaggle test split: engine size and age dominate; kilometres matter less once age is known."),
  H2("EV versus Petrol Depreciation (RQ3)"),
  P("Holding the variant and city fixed, a used electric two-wheeler loses 12.0% of its value per year against 6.3% for a petrol one — about twice as fast (Table VI, Fig. 4) — so the hypothesis is supported (p = 0.001). The gap holds without Ola, the largest EV brand, and among scooters alone, so it is not caused by one brand or by comparing scooters with motorcycles. A brand effect is not claimed: an earlier, smaller collection placed most of the gap on Ola, while the larger one does not."),
  ...TABLE("Same-variant fixed-effects regression, ads aged ≤ 7 years", ["Check", "Ads", "EV ads", "Petrol %/yr", "EV %/yr", "p"], [
    ["All two-wheelers", "20,014", "847", "−6.3", "−12.0", "0.001"],
    ["Without Ola", "19,627", "460", "−6.3", "−13.0", "0.002"],
    ["Ola only", "19,554", "387", "−6.3", "−10.6", "0.125"],
    ["Scooters only", "3,506", "717", "−7.6", "−12.9", "0.010"],
  ], [1240, 760, 660, 860, 760, 760], { boldRows: [0] }),
  ...FIGURE("m7_value_kept.png", 975, 600, "Asking price as a share of today's new price, by age (2026 BikeWale ads): EVs without Ola, Ola, and petrol two-wheelers."),
  H2("Does the Tax History Help?"),
  P("Adding the two tax inputs and retraining on the same split changes the typical miss from 12.41% to 12.37% for two-wheelers and from 8.28% to 8.32% for cars, and the share within ±20% by under 0.2 point — all within noise. The model year already tells the model what the tax rate would, because the rate is a function of the year. Unconstrained tax inputs also let an older vehicle score above a newer one in 489 of 1,104 models, because rates jump between years; the deployed system therefore holds each price at or below that of the next newer year. The record is kept for transparency: the application shows the tax a first owner paid and how it compares with today's rate."),
  ...FIGURE("m9b_tax_history.png", 1500, 630, "Effective national tax on a new vehicle in India, 1990–2026 (excise + VAT, then GST and GST 2.0)."),
  H2("Deployment"),
  P("The models are served by a FastAPI web application that prices 1,104 two-wheeler and car models in a few milliseconds, routes models without enough 2026 ads to a calibrated older model or a depreciation formula, and shows the price, its range, the SHAP reasons, a depreciation curve and the before- and after-tax price. An automated sweep prices every model at three ages and checks for errors, rising curves and gaps to similar ads; the final run reports no issues and a median gap of 5.3% to the median of similar ads."),

  H1("Discussion"),
  P("All three research questions are answered positively: XGBoost beats the base paper's algorithms, a 2020–21 model is biased on 2026 data, and electric two-wheelers depreciate about twice as fast as petrol ones. This agrees with car studies in the US [18], [19] and extends the finding to Indian two-wheelers. Applying only the measured yearly rates, a ₹1.2 lakh electric scooter would be worth about ₹72,000 after four years against about ₹93,000 for a petrol scooter with the same new price, so lenders and buy-back schemes should use a steeper curve for EVs. Likely causes, not tested here, are battery wear and the lack of battery-health data, falling new-EV prices and subsidies, and fast model updates."),
  P("The study has limits. Listings give asking prices, not sale prices. There is no condition, accident or battery-health data. The 2026 data are a one-week snapshot with 891 EV ads, so brand-level EV effects are unreliable. A random split can place ads of the same model in both training and test sets; this suits an application that prices known models, but scores for unseen models would be lower. Tax rates before 2008 are poorly documented."),

  H1("Conclusion"),
  P("We predicted resale prices of used two-wheelers and cars in India with monotone, explained XGBoost models trained on 2026 listings, reaching errors close to the market's own noise floor and beating a commercial valuation for cars. A model trained on older data went stale, and used electric two-wheelers lost value about twice as fast as petrol ones of the same variant. A 1990–2026 tax record did not improve accuracy but makes the original price transparent. Future work will repeat the collection monthly, add battery state-of-health from service records, and use transaction prices instead of asking prices."),
];

const refs = [
  "K. H. Ferdous, M. S. Arman, K. R. Ahmed, N. Tasnim, I. Mahmud and P. Revathi, \"Predicting pre-owned bike values with machine learning techniques in Bangladesh,\" in *Proc. Int. Conf. Data Science and Network Security (ICDSNS)*, Tiptur, India, 2024, doi: 10.1109/ICDSNS62112.2024.10691304.",
  "\"Interpretable machine learning for price prediction in India's used two-wheeler market using XAI tools,\" 2025.",
  "K. Sathyanarayanan, \"Pre-owned bike price prediction using machine learning,\" M.Sc. thesis, National College of Ireland, Dublin, 2023.",
  "\"Price prediction of used bike using machine learning,\" *Int. J. Scientific & Technical Development*, vol. 10, no. 1, 2024.",
  "\"Used car price prediction and feature importance analysis using XGBoost and SHAP,\" 2025.",
  "\"Explainable used car pricing: A machine learning framework with marginal effects,\" The University of Hong Kong, 2025.",
  "Luo *et al.*, \"Revolutionizing the Malaysian used car market,\" IEEE, 2023.",
  "Bukvić *et al.*, \"Price prediction and classification of used-vehicles using supervised machine learning,\" 2022.",
  "\"Car price prediction using machine learning algorithms,\" *Int. J. Research in Applied Science & Engineering Technology*, 2022.",
  "\"Prediction of used car prices using machine learning techniques,\" *Int. Research J. Engineering and Technology*, 2022.",
  "\"Price prediction for used cars: A comparison of machine learning regression models,\" degree project, KTH Royal Institute of Technology, 2022.",
  "\"Prediction of the price of used cars based on machine learning algorithms,\" 2023.",
  "\"Machine learning-based models for accurate car prices prediction,\" 2024.",
  "\"Machine learning optimization and challenges in used car price prediction,\" 2025.",
  "\"Car price prediction and recognition using deep learning and computer vision,\" 2025.",
  "\"Linear regression for used car price prediction,\" *Int. J. Computational and Experimental Science and Engineering*, 2023, doi: 10.22399/ijcesen.1070505.",
  "\"Used car price prediction using machine learning,\" 2023.",
  "C. Roberson, A. Pantha and J. P. Helveston, \"Battery-powered bargains? Assessing electric vehicle resale value in the United States,\" 2024.",
  "\"Depreciation dynamics of electric vs. internal combustion engine vehicles in the US car market,\" thesis, University of Michigan, 2025.",
  "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785–794.",
  "S. M. Lundberg *et al.*, \"From local explanations to global understanding with explainable AI for trees,\" *Nature Machine Intelligence*, vol. 2, pp. 56–67, 2020.",
  "saisaathvik, \"Used bikes prices in India,\" Kaggle dataset.",
  "N. Birla, \"Vehicle dataset from CarDekho,\" Kaggle dataset.",
  "BikeWale, used two-wheeler listings and model pages, bikewale.com, accessed Nov. 2026.",
  "CarWale, used car listings and model pages, carwale.com, accessed Nov. 2026.",
  "Ministry of Finance, Union Budget 2008–09: central excise changes; Central Excise notifications 2008–2010 and Budget 2012–13, indiabudget.gov.in.",
  "GST Council, 36th meeting (EV rate 12% to 5%, Aug. 2019) and 56th meeting (GST rate rationalisation, 22 Sep. 2025), Press Information Bureau.",
  "\"A taxing drive: excise duty on cars since 1986,\" *Business Today*, 2009.",
];
const refParas = [
  new Paragraph({ children: [new TextRun({ text: "References", font: FONT, size: 20, smallCaps: true })], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true }),
  ...refs.map((r, i) => new Paragraph({
    children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 16 }), ...runs(r, { size: 16 })],
    tabStops: [{ type: TabStopType.LEFT, position: 400 }], indent: { left: 400, hanging: 400 },
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 30, line: 200 } })),
];

const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900 } };
const doc = new Document({
  creator: "Kartik Katwe, Mohammed Fouzan Anwar, Sathvi Nagraj, Ishaan Nandi Roy",
  title: "Used Two-Wheeler Resale Price Prediction in India with Electric-Vehicle Depreciation Analysis",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [
    { properties: { page }, children: front },
    { properties: { page, type: SectionType.CONTINUOUS, column: { space: 360, count: 2 } }, children: [...body, ...refParas] },
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length); });
