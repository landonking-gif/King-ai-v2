---
name: python_exec
description: Execute Python code in an isolated environment for calculations, file generation, and data processing
version: 1.0.0
author: Architecture Team
license: MIT
dependencies:
  - python>=3.11
  - python-docx>=1.1.0
  - openpyxl>=3.1.0
  - pandas>=2.0.0
  - matplotlib>=3.8.0
  - reportlab>=4.0.0
  - Pillow>=10.0.0
tags:
  - execution
  - python
  - code
  - calculations
  - document-generation
  - data-processing
---

# Python Execution Skill

Execute Python code in an isolated subprocess environment with timeout protection. This skill is essential for:

- **Document Generation**: Create Word (.docx), Excel (.xlsx), PowerPoint (.pptx), and PDF files
- **Calculations**: Perform mathematical operations, statistical analysis, and financial computations
- **Data Processing**: Transform, aggregate, and analyze data from various sources
- **Research Steps**: Intermediate computations during multi-step research workflows

## When to Use

### Document Generation
Use this skill when you need to create binary document formats that cannot be generated with simple file writes:

- Word documents with formatted tables, headers, styles
- Excel spreadsheets with formulas, charts, multiple sheets
- PowerPoint presentations with slides, images, layouts
- PDF reports with styling, tables, embedded images

### Calculations & Analysis
Use this skill for computational tasks during query processing:

- Mathematical calculations (percentages, averages, sums, ratios)
- Financial metrics (CAGR, growth rates, margin calculations)
- Statistical analysis (mean, median, standard deviation)
- Date/time calculations and formatting
- Aggregating data from multiple sources

### Deep Research Steps
Use this skill as an intermediate step in research workflows:

- Process and validate extracted data from documents
- Compute comparisons between entities (e.g., ESG scores)
- Generate summary statistics from research findings
- Transform extracted JSON/CSV data into structured outputs
- Merge data from multiple document extractions

### Visualization
Use this skill to create visual representations of data:

- Charts and graphs with matplotlib or plotly
- Data visualizations for reports
- Infographics from analyzed data

## When NOT to Use

- Writing plain text or markdown files (use `file_write`)
- Reading files (use `file_read` or `document_qa`)
- Searching files (use `file_grep` or `file_glob`)
- Running shell commands (use `bash_exec`)
- Simple string manipulation
- Just displaying text output

## Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code` | string | Yes | - | Python code to execute |
| `timeout_seconds` | integer | No | 30 | Timeout in seconds (max 300) |

### Input Example

```json
{
  "code": "import pandas as pd\nfrom openpyxl import Workbook\n\n# Create Excel with data\nwb = Workbook()\nws = wb.active\nws['A1'] = 'Company'\nws['B1'] = 'ESG Score'\nws['A2'] = 'Tata Steel'\nws['B2'] = 85.5\nwb.save('/tmp/esg_report.xlsx')\nprint('Created: /tmp/esg_report.xlsx')",
  "timeout_seconds": 60
}
```

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `output` | string | Standard output from execution |
| `error` | string | Error message if execution failed |
| `return_value` | any | Return value from the last expression |
| `execution_time_ms` | number | Execution time in milliseconds |
| `success` | boolean | Whether execution succeeded |

### Output Example

```json
{
  "output": "Created: /tmp/esg_report.xlsx",
  "error": null,
  "return_value": null,
  "execution_time_ms": 245.67,
  "success": true
}
```

## Examples

### Create a Word Document

```python
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Add title
title = doc.add_heading('ESG Report: Tata Steel', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Add summary paragraph
doc.add_paragraph('This report summarizes the key ESG performance indicators...')

# Add table
table = doc.add_table(rows=4, cols=3)
table.style = 'Table Grid'
headers = table.rows[0].cells
headers[0].text = 'Category'
headers[1].text = 'Score'
headers[2].text = 'Trend'

data = [
    ('Environmental', '82.5', 'Improving'),
    ('Social', '78.3', 'Stable'),
    ('Governance', '91.2', 'Improving'),
]

for i, (cat, score, trend) in enumerate(data, 1):
    row = table.rows[i].cells
    row[0].text = cat
    row[1].text = score
    row[2].text = trend

doc.save('/Users/output/esg_report.docx')
print('Document created successfully')
```

### Calculate Financial Metrics

```python
import json

# Data extracted from documents
data = {
    'revenue_2022': 150000000,
    'revenue_2023': 175000000,
    'costs_2023': 140000000,
    'total_assets': 500000000,
    'total_liabilities': 200000000,
}

# Calculate metrics
revenue_growth = ((data['revenue_2023'] - data['revenue_2022']) / data['revenue_2022']) * 100
profit_margin = ((data['revenue_2023'] - data['costs_2023']) / data['revenue_2023']) * 100
debt_ratio = (data['total_liabilities'] / data['total_assets']) * 100

results = {
    'revenue_growth_pct': round(revenue_growth, 2),
    'profit_margin_pct': round(profit_margin, 2),
    'debt_ratio_pct': round(debt_ratio, 2),
}

print(json.dumps(results, indent=2))
```

### Generate Chart from Data

```python
import matplotlib.pyplot as plt

companies = ['Tata Steel', 'JSW Steel', 'SAIL', 'Hindalco']
esg_scores = [85.5, 78.2, 72.1, 81.3]

plt.figure(figsize=(10, 6))
bars = plt.bar(companies, esg_scores, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
plt.xlabel('Company')
plt.ylabel('ESG Score')
plt.title('ESG Score Comparison - Indian Steel Companies')
plt.ylim(0, 100)

for bar, score in zip(bars, esg_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{score}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/esg_comparison.png', dpi=150)
print('Chart saved to /tmp/esg_comparison.png')
```

### Aggregate Research Data

```python
import json

# Extracted data from multiple documents
tata_data = {'env': 82, 'social': 78, 'gov': 91}
jsw_data = {'env': 75, 'social': 80, 'gov': 85}
sail_data = {'env': 68, 'social': 72, 'gov': 78}

# Aggregate and compare
companies = {
    'Tata Steel': tata_data,
    'JSW Steel': jsw_data,
    'SAIL': sail_data,
}

summary = []
for name, scores in companies.items():
    avg = sum(scores.values()) / len(scores)
    summary.append({
        'company': name,
        'environmental': scores['env'],
        'social': scores['social'],
        'governance': scores['gov'],
        'overall_avg': round(avg, 1),
    })

# Sort by overall average
summary.sort(key=lambda x: x['overall_avg'], reverse=True)

print('ESG Rankings:')
for i, company in enumerate(summary, 1):
    print(f"{i}. {company['company']}: {company['overall_avg']}")

print('\nDetailed Results:')
print(json.dumps(summary, indent=2))
```

## Available Libraries

The following libraries are pre-installed and available:

| Category | Libraries |
|----------|-----------|
| Documents | `python-docx`, `openpyxl`, `python-pptx`, `reportlab`, `fpdf` |
| Data | `pandas`, `numpy`, `json`, `csv` |
| Visualization | `matplotlib`, `plotly`, `seaborn` |
| Images | `Pillow`, `cairosvg` |
| Utilities | `datetime`, `pathlib`, `tempfile` |

## Safety & Limitations

- **Timeout**: Maximum execution time is 300 seconds (5 minutes)
- **Output Limit**: Output is truncated at 30,000 characters
- **Isolation**: Code runs in a subprocess for isolation
- **Approval**: This skill requires policy approval before execution
- **No Network**: External network calls should use dedicated skills

## Best Practices

1. **Print outputs**: Always print results so they appear in the output
2. **Save files to accessible paths**: Use paths the user can access
3. **Handle errors gracefully**: Use try/except for robust execution
4. **Use appropriate libraries**: Choose the right library for the task
5. **Keep code focused**: One clear objective per execution
