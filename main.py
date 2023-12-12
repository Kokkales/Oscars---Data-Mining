from datapre import DataPreprocessor
from classifications import Classificationer

def seperateData(ds):
  target=ds['oscar winners']
  col=[]
  for feature in ds.columns:
    if feature!='oscar winners':
      col.append(feature)
  data=ds[col]
  return (target,data)


# preprocessing
dp=DataPreprocessor("Book.xlsx","datesThree.xlsx")
# print(dp.executePreprocess())
dataset=dp.executePreprocess()

# features,target
data,target=seperateData(dataset)
print(data.shape)
print(target.shape)


# classification
cl=Classificationer()
print(cl.executeSvmClassification())

print("Hey I am main!")