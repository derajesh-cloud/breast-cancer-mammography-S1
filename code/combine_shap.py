from PIL import Image
a=Image.open('/home/claude/work/figs/fig_shap_C1.png'); b=Image.open('/home/claude/work/figs/fig_shap_C3.png')
h=max(a.height,b.height); c=Image.new('RGB',(a.width+b.width,h),'white'); c.paste(a,(0,0)); c.paste(b,(a.width,0))
c.save('/home/claude/work/figs/fig_shap_combined.png',dpi=(300,300)); print(c.size)
