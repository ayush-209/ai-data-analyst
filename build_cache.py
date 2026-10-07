"""Seeds demo_cache.json with the planned demo questions so Offline demo mode works with no internet.
Live answers you get while rehearsing are added to the same file automatically."""
import json
from pathlib import Path

QUESTIONS = {
    "What are the total sales and profit by region?": '''
d = df.copy()
result = d.groupby('Region', dropna=False)[['Sales', 'Profit']].sum().round(0).sort_values('Sales', ascending=False)
fig = px.bar(result.reset_index(), x='Region', y=['Sales', 'Profit'], barmode='group',
             title='Total Sales and Profit by Region', labels={'value': 'Amount (₹)', 'variable': 'Metric'})
top = result.index[0]
summary = f"{top} leads with ₹{result.loc[top, 'Sales']:,.0f} in sales and ₹{result.loc[top, 'Profit']:,.0f} in profit across {len(result)} region labels."
''',
    "Show the monthly sales trend": '''
d = df.copy()
d['Order_Date'] = pd.to_datetime(d['Order_Date'], dayfirst=True, errors='coerce')
result = d.set_index('Order_Date').resample('MS')['Sales'].sum().round(0).to_frame('Sales')
fig = px.line(result.reset_index(), x='Order_Date', y='Sales', markers=True,
              title='Monthly Sales Trend', labels={'Order_Date': 'Month', 'Sales': 'Sales (₹)'})
peak = result['Sales'].idxmax()
summary = f"Sales peak in {peak:%B %Y} at ₹{result['Sales'].max():,.0f}, with a clear spike every Q4 (Oct–Dec)."
''',
    "Why is profit negative in Furniture?": '''
d = df[df['Category'] == 'Furniture'].copy()
result = d.groupby('Sub_Category').agg(Total_Sales=('Sales', 'sum'), Total_Profit=('Profit', 'sum'),
                                       Avg_Discount=('Discount', 'mean'), Orders=('Order_ID', 'count')).round(2)
result['Profit_Margin_%'] = (result['Total_Profit'] / result['Total_Sales'] * 100).round(1)
result = result.sort_values('Total_Profit')
fig = px.scatter(result.reset_index(), x='Avg_Discount', y='Profit_Margin_%', size='Total_Sales', text='Sub_Category',
                 title='Furniture: Discount vs Profit Margin', labels={'Avg_Discount': 'Average discount', 'Profit_Margin_%': 'Profit margin (%)'})
fig.update_traces(textposition='top center')
fig.add_hline(y=0, line_dash='dash')
worst = result.index[0]
summary = (f"Furniture's losses come from {worst}: it loses ₹{abs(result.loc[worst, 'Total_Profit']):,.0f} at an average "
           f"{result.loc[worst, 'Avg_Discount']:.0%} discount, while other furniture lines with lower discounts stay profitable.")
''',
    "Which region is the best?": '''
d = df.copy()
result = d.groupby('Region')['Sales'].sum().round(0).sort_values(ascending=False).to_frame('Total_Sales')
fig = px.bar(result.reset_index(), x='Region', y='Total_Sales', title='Best Region by Total Sales')
summary = f"{result.index[0]} is the best region with ₹{result['Total_Sales'].iloc[0]:,.0f} in total sales."
''',
    "Which region is the best by profit margin?": '''
d = df[df['Region'].notna() & (df['Region'] != 'Unknown')].copy()
result = d.groupby('Region').agg(Orders=('Order_ID', 'count'), Total_Sales=('Sales', 'sum'), Total_Profit=('Profit', 'sum'))
result['Margin_%'] = (result['Total_Profit'] / result['Total_Sales'] * 100).round(1)
result = result.round({'Total_Sales': 0, 'Total_Profit': 0}).sort_values('Margin_%', ascending=False)
fig = px.bar(result.reset_index(), x='Region', y='Margin_%', title='Profit Margin by Region',
             labels={'Margin_%': 'Profit margin (%)'})
top_sales = result['Total_Sales'].idxmax()
summary = (f"By profit margin, {result.index[0]} is best at {result['Margin_%'].iloc[0]:.1f}%, while {top_sales}, the top region "
           f"by total sales, earns {result.loc[top_sales, 'Margin_%']:.1f}%. 'Best' depends on the metric you choose.")
''',
    "Which customer segment gives the highest profit margin?": '''
d = df[df['Customer_Segment'].notna() & (df['Customer_Segment'] != 'Unknown')].copy()
result = d.groupby('Customer_Segment')[['Sales', 'Profit']].sum()
result['Margin_%'] = (result['Profit'] / result['Sales'] * 100).round(1)
result = result.round({'Sales': 0, 'Profit': 0}).sort_values('Margin_%', ascending=False)
fig = px.bar(result.reset_index(), x='Customer_Segment', y='Margin_%', title='Profit Margin by Customer Segment',
             labels={'Margin_%': 'Profit margin (%)', 'Customer_Segment': 'Segment'})
summary = f"{result.index[0]} has the highest profit margin at {result['Margin_%'].iloc[0]:.1f}%."
''',
}


def norm(q):
    import re
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


path = Path(__file__).parent / "demo_cache.json"
cache = json.loads(path.read_text()) if path.exists() else {}
for q, code in QUESTIONS.items():
    cache[norm(q)] = {"question": q, "code": code.strip()}
path.write_text(json.dumps(cache, indent=2))
print(f"demo_cache.json now holds {len(cache)} questions")
