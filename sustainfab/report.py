"""Self-contained PDF case-study reporting; no fonts/assets downloaded."""
from __future__ import annotations
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String
from .advanced import Economics, MaterialImpact, MaterialAssumptions, UncertaintyResult
from .model import SimulationResult

def _bars(baseline: float, selected: float) -> Drawing:
    chart = Drawing(445, 52)
    highest = max(baseline, selected, 1e-9)
    for i, (caption, value, shade) in enumerate((
        ("Baseline", baseline, colors.HexColor("#8496AA")),
        ("Selected", selected, colors.HexColor("#15988F")),
    )):
        y = 28 - 24 * i
        chart.add(String(2, y + 3, caption, fontSize=9))
        chart.add(Rect(77, y, max(0, value) / highest * 280, 15,
                       strokeColor=None, fillColor=shade))
        chart.add(String(376, y + 3, f"{value:,.1f}", fontSize=9))
    return chart

def build_pdf_report(
    *, baseline: SimulationResult, selected: SimulationResult,
    economics: Economics, baseline_material: MaterialImpact,
    selected_material: MaterialImpact, material_assumptions: MaterialAssumptions,
    baseline_uncertainty: UncertaintyResult, selected_uncertainty: UncertaintyResult,
    currency: str, usd_to_currency: float, fx_updated: str,
    usd_tariff: float, demand: float, operating_hours: float,
    assumptions_description: str,
) -> bytes:
    """Portable report based on the exact calculated dashboard scenario."""
    out = BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm,
                            title="SustainFab engineering case study")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SFTitle", parent=styles["Title"],
                               textColor=colors.HexColor("#158880"),
                               fontSize=20, leading=24, spaceAfter=9))
    styles.add(ParagraphStyle(name="SFSmall", parent=styles["Normal"],
                               fontSize=8, leading=11, spaceAfter=6))
    styles.add(ParagraphStyle(name="SFHead", parent=styles["Heading3"],
                               textColor=colors.HexColor("#117B75"), spaceBefore=10))
    story = [
        Paragraph("SustainFab | Engineering Case Study", styles["SFTitle"]),
        Paragraph("Sustainable Manufacturing Systems Optimizer", styles["Heading3"]),
        Paragraph("Illustrative, synthetic engineering scenario; not calibrated industrial evidence.", styles["SFSmall"]),
        Paragraph(f"Scheduled {operating_hours:.2f} h/day | Demand {demand:,.1f} good units/day | "
                  f"Reference electricity tariff USD {usd_tariff:.4f}/kWh", styles["Normal"]),
    ]
    def header(t):
        story.append(Paragraph(t, styles["SFHead"]))
    def comparison(rows):
        t = Table(rows, colWidths=[94*mm, 40*mm, 40*mm], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DBF0ED")),
            ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#B7D6D2")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t)
    header("1. Deterministic line performance")
    comparison([
        ["Metric", "Baseline", "Selected"],
        ["Good output (units/day)", f"{baseline.good_units:,.1f}", f"{selected.good_units:,.1f}"],
        ["Energy (kWh/day)", f"{baseline.total_energy_kwh:,.1f}", f"{selected.total_energy_kwh:,.1f}"],
        ["Energy intensity (kWh/good unit)", f"{baseline.kwh_per_good_unit or 0:.3f}", f"{selected.kwh_per_good_unit or 0:.3f}"],
        ["Cost ("+currency+"/day)", f"{baseline.electricity_cost:,.2f}", f"{selected.electricity_cost:,.2f}"],
        ["Operational CO2 (kg/day)", f"{baseline.emissions_kg:,.1f}", f"{selected.emissions_kg:,.1f}"],
    ])
    header("2. Output and energy comparison")
    story.append(Paragraph("Good output (units/day)", styles["SFSmall"]))
    story.append(_bars(baseline.good_units, selected.good_units))
    story.append(Paragraph("Electricity consumed (kWh/day)", styles["SFSmall"]))
    story.append(_bars(baseline.total_energy_kwh, selected.total_energy_kwh))
    header("3. Techno-economic assessment")
    payback = (f"{economics.simple_payback_years:.2f} years"
               if economics.simple_payback_years is not None else "not applicable")
    story.append(Paragraph(
        f"Additional machines: {economics.added_machines}; incremental investment: "
        f"USD {economics.investment_usd:,.2f}; annual net benefit: "
        f"USD {economics.annual_net_benefit_usd:,.2f}; "
        f"NPV: USD {economics.npv_usd:,.2f}; simple payback: {payback}.", styles["Normal"]))
    story.append(Paragraph(assumptions_description, styles["SFSmall"]))
    header("4. Material inputs and scrap")
    comparison([
        ["Material indicator", "Baseline", "Selected"],
        ["Input feedstock (kg/day)", f"{baseline_material.input_material_kg:,.1f}",
         f"{selected_material.input_material_kg:,.1f}"],
        ["Scrap (kg/day)", f"{baseline_material.scrap_kg:,.1f}",
         f"{selected_material.scrap_kg:,.1f}"],
        ["Unrecovered scrap (kg/day)", f"{baseline_material.unrecovered_kg:,.1f}",
         f"{selected_material.unrecovered_kg:,.1f}"],
        ["Net material CO2e (kg/day)", f"{baseline_material.net_material_kgco2e:,.1f}",
         f"{selected_material.net_material_kgco2e:,.1f}"],
    ])
    story.append(Paragraph(
        f"Material assumptions: {material_assumptions.mass_per_input_unit_kg:g} kg/input unit; "
        f"virgin factor {material_assumptions.embodied_kgco2e_per_kg:g} kg CO2e/kg; "
        f"recovered scrap {material_assumptions.recovered_scrap_fraction:.0%}; "
        f"potential credit {material_assumptions.recovery_credit_kgco2e_per_kg:g} kg CO2e/kg.",
        styles["SFSmall"]))
    header("5. Monte Carlo uncertainty")
    comparison([
        ["Good-output indicator", "Baseline", "Selected"],
        ["Mean (units/day)", f"{baseline_uncertainty.mean_good_units:.1f}",
         f"{selected_uncertainty.mean_good_units:.1f}"],
        ["5th-95th pct (units/day)", f"{baseline_uncertainty.p05_good_units:.1f}-{baseline_uncertainty.p95_good_units:.1f}",
         f"{selected_uncertainty.p05_good_units:.1f}-{selected_uncertainty.p95_good_units:.1f}"],
        ["Pr(meeting selected target)", f"{baseline_uncertainty.target_meeting_probability:.1%}",
         f"{selected_uncertainty.target_meeting_probability:.1%}"],
    ])
    story.append(Paragraph(
        f"Draws: {baseline_uncertainty.samples}; seed: {baseline_uncertainty.seed}; "
        f"target: {baseline_uncertainty.target_good_units:.1f} units/day.", styles["SFSmall"]))
    header("6. Reproducibility and limitations")
    story.append(Paragraph(
        f"Exchange reference: 1 USD = {usd_to_currency:,.4f} {currency}; "
        f"FX data last updated: {fx_updated}. Quoted rates are indicative.", styles["SFSmall"]))
    story.append(Paragraph(
        "This steady-state screening model omits queues, buffers, changeovers, "
        "stochastic event-level failure/repair dynamics, embodied capital-equipment impacts, "
        "labor, financing and tax. Monte Carlo uses assumed independent clipped-normal "
        "daily parameter fluctuations, not measured probabilities. Material results use "
        "constant part mass and hypothetical recycling displacement credits. "
        "Calibrate all coefficients with measured data before plant investment or "
        "environmental decisions.", styles["SFSmall"]))
    def footer(canvas, template):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(18*mm, 10*mm, "SustainFab - illustrative engineering assessment")
        canvas.drawRightString(A4[0]-18*mm, 10*mm, f"Page {template.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
