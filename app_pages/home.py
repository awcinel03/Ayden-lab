import streamlit as st

from lab.registry import Category, EntryType, published_entries
from lab.ui import empty_state, entry_list

st.title("Ayden's Lab")
st.write(
    "Projects, tools, findings and experiments by Ayden Cinel, across network science, "
    "machine learning, the Claude API, model testing and AI more broadly. "
    "Every result links back to its data and the code that produced it."
)

entries = published_entries()

st.subheader("Recent entries")
categories = st.pills(
    "Category", list(Category), selection_mode="multi", format_func=lambda c: c.value
)
types = st.pills("Type", list(EntryType), selection_mode="multi", format_func=lambda t: t.value)
all_tags = sorted({tag for e in entries for tag in e.tags})
tags = st.multiselect("Tags", all_tags, disabled=not all_tags)

shown = [
    e
    for e in entries
    if (not categories or e.category in categories)
    and (not types or e.type in types)
    and (not tags or set(tags) & set(e.tags))
]

if not entries:
    empty_state("No entries yet. The first experiments are on their way.")
elif not shown:
    empty_state("No entries match these filters.")
else:
    entry_list(shown)
