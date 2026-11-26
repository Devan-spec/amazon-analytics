ElevanceSkills Amazon Product Analytics Project

Files:
- amazon_product_analytics.py : complete script for all six assignment tasks
- requirements.txt             : Python dependencies
- amazon.csv                   : put your supplied dataset in this folder (or pass its path)

Ubuntu setup:
1) Create and activate a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate
2) Install dependencies:
   pip install -r requirements.txt
3) Run:
   python amazon_product_analytics.py --input amazon.csv --output outputs

The script writes cleaned data, CSV result tables, graphs, regression summaries,
and a saved best model into the outputs folder.

Important methodology notes:
- Duplicate product IDs are aggregated to one product row: numeric values use median,
  category uses mode, and review text is concatenated for sentiment analysis.
- The supplied data has 1,465 rows and 1,351 distinct product IDs. After required
  value validation, the script currently retains 1,348 product-level rows.
- Rating count is used as a popularity/demand proxy, not as sales or profit.
- All conclusions must be written after inspecting the generated outputs. The script
  does not invent business conclusions from metrics.
