import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("Model ve şema yükleniyor...")
model = joblib.load('ieee_fraud_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Modelden en önemli 15 özelliğin çekilmesi
importance = model.feature_importances_
df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15)

# Görsel tasarım
plt.figure(figsize=(10, 6))
sns.set_style("white") 

ax = sns.barplot(x='Importance', y='Feature', data=df_imp, color='#2c3e50')

sns.despine(left=True, bottom=True)
ax.set(xticklabels=[])  
ax.tick_params(bottom=False, left=False)
# Başlık ve eksen ayarları
plt.title('Top 15 Most Important Features for Fraud Detection', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('')
plt.xlabel('')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, transparent=True)
print("feature_importance.png başarıyla oluşturuldu!")
