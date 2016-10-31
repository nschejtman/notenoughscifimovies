import ItemCB
import time

est = ItemCB.ItemCB()
urm = ItemCB.read_interactions()
urm = urm[0:4001, :]
est.fit(urm, None, 5, ItemCB.items_df)

for i in range(10):
    est.sim(i,i)
