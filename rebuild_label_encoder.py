import pickle
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("./patient_model")
id2label = model.config.id2label

labels = [id2label[i] for i in sorted(id2label.keys())]
print(f"Found {len(labels)} labels:")
for i, label in enumerate(labels):
    print(f"  {i}: {label}")

le = LabelEncoder()
le.fit(labels)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("label_encoder.pkl rebuilt successfully!")
