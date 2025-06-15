import streamlit as st


st.markdown('''

###### Mushroom Attribute Codes and Definitions

| Attribute | Codes | Definition |
|-----------|-------|------------|
| Population | a, c, n, s, v, y | a: Abundant; c: Clustered; n: Numerous; s: Scattered; v: Several; y: Solitary |
| Cap-shape | b, c, x, f, k, s | b: Bell; c: Conical; x: Convex; f: Flat; k: Knobbed; s: Sunken |
| Cap-color | n, b, c, g, r, p, u, e, w, y | n: Brown; b: Buff; c: Cinnamon; g: Gray; r: Green; p: Pink; u: Purple; e: Red; w: White; y: Yellow |
| Bruises? | t, f | t: Bruises; f: No |
| Odor | a, l, c, y, f, m, n, p, s | a: Almond; l: Anise; c: Creosote; y: Fishy; f: Foul; m: Musty; n: None; p: Pungent; s: Spicy |
| Gill-attachment | a, d, f, n | a: Attached; d: Descending; f: Free; n: Notched |
| Gill-size | b, n | b: Broad; n: Narrow |
| Gill-color | k, n, b, h, g | k: Black; n: Brown; b: Buff; h: Chocolate; g: Gray |
| Spore-print-color | k, n, b, h, r, o, u, w, y | k: Black; n: Brown; b: Buff; h: Chocolate; r: Green; o: Orange; u: Purple; w: White; y: Yellow |
| Habitat | g, l, m, p, u, w, d | g: Grasses; l: Leaves; m: Meadows; p: Paths; u: Urban; w: Waste; d: Woods |

            ''')