import re

with open("streamlit_app.py", "r") as f:
    text = f.read()

# Replace all plain #000000 with off-white for data visualization and basic spans
text = text.replace('color:#000000', 'color:#ecfdf5')
text = text.replace('color: #000000', 'color: #ecfdf5')
text = text.replace('color="#000000"', 'color="#cbd5e1"')

# Sidebar elements formatting
text = text.replace('color:#ecfdf5; letter-spacing:-0.5px;">AgroAgent', 'color:#a7f3d0; letter-spacing:-0.5px;">AgroAgent')
text = text.replace('color:#ecfdf5; margin-top:2px;">Smart Crop Advisory', 'color:#6ee7b7; margin-top:2px;">Smart Crop Advisory')
text = text.replace('color:#ecfdf5; text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:6px;">Powered By', 'color:#6ee7b7; text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:6px;">Powered By')
text = text.replace('color:#ecfdf5; line-height:1.6;', 'color:#94a3b8; line-height:1.6;')

# Crop specific headings
text = text.replace('color:#ecfdf5;text-transform:uppercase;letter-spacing:2px;font-weight:600;">Recommended Crop', 'color:#a7f3d0;text-transform:uppercase;letter-spacing:2px;font-weight:600;">Recommended Crop')
text = text.replace('font-weight:700;color:#ecfdf5;">{rank_label}', 'font-weight:700;color:#6ee7b7;">{rank_label}')
text = text.replace('font-size:0.85rem;color:#ecfdf5;">{conf_pct:.1f}%', 'font-size:0.85rem;color:#6ee7b7;">{conf_pct:.1f}%')

# Market insights text
text = text.replace('style="font-size:0.9rem;color:#ecfdf5;">', 'style="font-size:0.9rem;color:#94a3b8;">')
text = text.replace('strong style="color:#ecfdf5;">Market Insights', 'strong style="color:#a7f3d0;">Market Insights')

# Plotly dict replacements to mute the chart lines a bit
text = text.replace('font=dict(color="#cbd5e1")', 'font=dict(color="#94a3b8")')

with open("streamlit_app.py", "w") as f:
    f.write(text)
