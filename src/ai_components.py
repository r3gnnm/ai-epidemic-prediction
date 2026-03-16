# ============================================================================
# 🤖 AI-ENHANCED EPIDEMIC PREDICTION SYSTEM
# Готовый код для хакатона
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
import anthropic

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("🤖 AI-ENHANCED EPIDEMIC PREDICTION SYSTEM")
print("="*70)

# ============================================================================
# КОМПОНЕНТ 1: ML-ПРЕДСКАЗАНИЕ ВСПЫШЕК
# ============================================================================

class OutbreakPredictor:
    """
    Machine Learning модель для предсказания вспышек эпидемий
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    def create_synthetic_training_data(self, vaccination_plan, n_samples=500):
        """
        Создаёт синтетические данные для обучения
        (для хакатона, когда нет реальных исторических данных)
        
        Логика:
        - Зоны с высоким Comfort Index → вспышка с высокой вероятностью
        - Зоны с низким Comfort Index → вспышка с низкой вероятностью
        - Добавляем шум для реалистичности
        """
        
        print("\n🔧 Creating synthetic training data...")
        
        data = []
        
        # Генерируем примеры на основе текущих зон
        for _ in range(n_samples):
            # Случайная зона как базис
            zone = vaccination_plan[np.random.randint(0, len(vaccination_plan))]
            
            # Копируем данные с небольшими вариациями
            temp = zone['satellite_data']['temperature'] + np.random.normal(0, 2)
            humidity = zone['satellite_data']['humidity'] + np.random.normal(0, 5)
            ndvi = np.clip(zone['satellite_data']['ndvi'] + np.random.normal(0, 0.1), -1, 1)
            ndwi = np.clip(zone['satellite_data']['ndwi'] + np.random.normal(0, 0.1), -1, 1)
            precip = max(0, zone['satellite_data']['precipitation_30d'] + np.random.normal(0, 20))
            
            # Рассчитываем comfort на основе вариаций
            comfort = zone['bacteria_comfort']['overall_comfort'] + np.random.normal(0, 10)
            comfort = np.clip(comfort, 0, 100)
            
            # Вероятность вспышки зависит от comfort index
            # Высокий comfort → высокая вероятность вспышки
            outbreak_probability = comfort / 100 + np.random.normal(0, 0.15)
            outbreak = 1 if outbreak_probability > 0.5 else 0
            
            data.append({
                'temperature': temp,
                'humidity': humidity,
                'ndvi': ndvi,
                'ndwi': ndwi,
                'precipitation_30d': precip,
                'comfort_index': comfort,
                'month': np.random.randint(1, 13),
                'is_rainy_season': np.random.choice([0, 1]),
                'population_density': zone['population'] / 1000 + np.random.normal(0, 10),
                'outbreak': outbreak
            })
        
        df = pd.DataFrame(data)
        
        print(f"✅ Created {len(df)} training samples")
        print(f"   Outbreaks: {df['outbreak'].sum()} ({df['outbreak'].mean()*100:.1f}%)")
        print(f"   No outbreaks: {(1-df['outbreak']).sum()} ({(1-df['outbreak']).mean()*100:.1f}%)")
        
        return df
    
    def train(self, training_data):
        """
        Обучает ML-модель
        """
        
        print("\n🤖 Training ML model...")
        
        # Разделяем на X и y
        feature_columns = [col for col in training_data.columns if col != 'outbreak']
        X = training_data[feature_columns]
        y = training_data['outbreak']
        
        self.feature_names = feature_columns
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Обучаем Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced',  # для несбалансированных данных
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Оценка
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n📊 Model Performance on Test Set:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['No Outbreak', 'Outbreak']))
        
        # ROC-AUC
        auc = roc_auc_score(y_test, y_proba)
        print(f"🎯 ROC-AUC Score: {auc:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"              No    |   Yes")
        print(f"Actual No   {cm[0,0]:4d}  |  {cm[0,1]:4d}")
        print(f"Actual Yes  {cm[1,0]:4d}  |  {cm[1,1]:4d}")
        
        self.is_trained = True
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Feature Importance:")
        for idx, row in importances.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return importances
    
    def predict(self, zone_data):
        """
        Предсказывает вероятность вспышки для зоны
        """
        
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        # Подготавливаем features
        features = pd.DataFrame([{
            'temperature': zone_data['satellite_data']['temperature'],
            'humidity': zone_data['satellite_data']['humidity'],
            'ndvi': zone_data['satellite_data']['ndvi'],
            'ndwi': zone_data['satellite_data']['ndwi'],
            'precipitation_30d': zone_data['satellite_data']['precipitation_30d'],
            'comfort_index': zone_data['bacteria_comfort']['overall_comfort'],
            'month': datetime.now().month,
            'is_rainy_season': 1 if datetime.now().month in [6,7,8,9,10] else 0,
            'population_density': zone_data['population'] / 1000,
        }])
        
        # Предсказание
        proba = self.model.predict_proba(features)[0, 1]
        
        return proba * 100  # в проценты
    
    def save(self, filepath='outbreak_ml_model.pkl'):
        """Сохраняет модель"""
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved: {filepath}")
    
    def load(self, filepath='outbreak_ml_model.pkl'):
        """Загружает модель"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"✅ Model loaded: {filepath}")


def add_ml_predictions_to_zones(vaccination_plan, predictor):
    """
    Добавляет ML-предсказания ко всем зонам
    """
    
    print("\n🤖 Adding ML predictions to all zones...")
    
    for zone in vaccination_plan:
        ml_proba = predictor.predict(zone)
        
        zone['ml_outbreak_probability'] = ml_proba
        
        # Классификация
        if ml_proba >= 75:
            zone['ml_risk'] = 'CRITICAL'
            zone['ml_color'] = '#d32f2f'
        elif ml_proba >= 50:
            zone['ml_risk'] = 'HIGH'
            zone['ml_color'] = '#f57c00'
        elif ml_proba >= 25:
            zone['ml_risk'] = 'MEDIUM'
            zone['ml_color'] = '#fbc02d'
        else:
            zone['ml_risk'] = 'LOW'
            zone['ml_color'] = '#689f38'
    
    print("✅ ML predictions added")
    
    return vaccination_plan


# ============================================================================
# КОМПОНЕНТ 2: AI-АССИСТЕНТ (CLAUDE)
# ============================================================================

class AIEpidemicAssistant:
    """
    LLM-ассистент для анализа эпидемиологических данных
    """
    
    def __init__(self, vaccination_plan, disease_profile, summary):
        # Примечание: для хакатона нужен API ключ Claude
        # Получить здесь: https://console.anthropic.com/
        try:
            self.client = anthropic.Anthropic()
            self.has_api = True
        except:
            print("⚠️ Claude API not available. Install: pip install anthropic")
            self.has_api = False
        
        self.vaccination_plan = vaccination_plan
        self.disease = disease_profile
        self.summary = summary
        self.context = self._prepare_context()
    
    def _prepare_context(self):
        """Подготавливает контекст для AI"""
        
        context = f"""
# EPIDEMIC PREDICTION SYSTEM DATA

## Disease: {self.disease['name']}
- Pathogen: {self.disease['pathogen']}
- R₀: {self.disease['R0']}
- Fatality Rate: {self.disease['fatality_rate']:.1%}
- Vaccine Efficacy: {self.disease['vaccine_efficacy']:.0%}

## Summary Statistics:
- Zones Analyzed: {self.summary['total_zones']}
- Population at Risk: {self.summary['total_population']:,}
- Lives Saved by Vaccination: {self.summary['prevented_deaths']:,}
- Total Cost: ${self.summary['total_cost_usd']:,.0f}
- Cost per Life: ${self.summary['cost_per_prevented_death']:,.0f}

## Top Zones by Risk:
"""
        
        # Топ-5 зон
        sorted_zones = sorted(self.vaccination_plan, 
                            key=lambda x: x.get('ml_outbreak_probability', 
                                               x['bacteria_comfort']['overall_comfort']), 
                            reverse=True)[:5]
        
        for i, zone in enumerate(sorted_zones, 1):
            ml_info = ""
            if 'ml_outbreak_probability' in zone:
                ml_info = f"\n   - ML Outbreak Probability: {zone['ml_outbreak_probability']:.1f}%"
            
            context += f"""
{i}. Zone #{zone['zone_number']} ({zone['priority']})
   - Bacteria Comfort: {zone['bacteria_comfort']['overall_comfort']:.1f}%{ml_info}
   - Lives Saved: {zone.get('lives_saved', 0):,}
   - Temperature: {zone['satellite_data']['temperature']}°C
   - Humidity: {zone['satellite_data']['humidity']}%
"""
        
        return context
    
    def ask(self, question):
        """Задать вопрос AI"""
        
        if not self.has_api:
            return "❌ Claude API not available. Please install anthropic and set API key."
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system="You are an expert epidemiologist AI. Answer clearly and cite data.",
                messages=[
                    {"role": "user", "content": f"{self.context}\n\nQuestion: {question}"}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"❌ API Error: {str(e)}"
    
    def generate_executive_summary(self):
        """Генерирует краткий отчёт для руководства"""
        
        prompt = """Generate a 1-page executive summary for government officials.
Include:
- Key risk zones
- Recommended actions
- Expected impact (lives saved)
- Budget needed
- Timeline

Use clear, non-technical language."""
        
        return self.ask(prompt)
    
    def explain_ml_prediction(self, zone_number):
        """Объясняет ML-предсказание для зоны"""
        
        zone = next((z for z in self.vaccination_plan 
                    if z['zone_number'] == zone_number), None)
        
        if not zone:
            return f"Zone #{zone_number} not found"
        
        if 'ml_outbreak_probability' not in zone:
            return "ML predictions not available for this zone"
        
        prompt = f"""Zone #{zone_number} has:
- ML Outbreak Probability: {zone['ml_outbreak_probability']:.1f}%
- Bacteria Comfort Index: {zone['bacteria_comfort']['overall_comfort']:.1f}%
- Temperature: {zone['satellite_data']['temperature']}°C
- Humidity: {zone['satellite_data']['humidity']}%
- NDWI: {zone['satellite_data']['ndwi']:.3f}

Explain in simple terms why ML predicted {'high' if zone['ml_outbreak_probability'] >= 50 else 'low'} risk."""
        
        return self.ask(prompt)


# ============================================================================
# ВИЗУАЛИЗАЦИЯ AI-РЕЗУЛЬТАТОВ
# ============================================================================

def visualize_ai_enhancements(vaccination_plan, ml_importances):
    """
    Создаёт графики AI-компонентов для демо
    """
    
    print("\n📊 Creating AI visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. ML Feature Importance
    ax1 = fig.add_subplot(gs[0, 0])
    top_features = ml_importances.head(8)
    bars = ax1.barh(range(len(top_features)), top_features['importance'], 
                    color='#4caf50', alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.set_xlabel('Importance', fontweight='bold', fontsize=11)
    ax1.set_title('🤖 ML: Most Important Features', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. ML Probability Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ml_probs = [z.get('ml_outbreak_probability', 0) for z in vaccination_plan]
    ax2.hist(ml_probs, bins=15, color='#2196f3', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax2.set_xlabel('ML Outbreak Probability (%)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Number of Zones', fontweight='bold', fontsize=11)
    ax2.set_title('🤖 ML: Probability Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. ML vs Comfort Index
    ax3 = fig.add_subplot(gs[0, 2])
    comfort_vals = [z['bacteria_comfort']['overall_comfort'] for z in vaccination_plan]
    ml_vals = [z.get('ml_outbreak_probability', 0) for z in vaccination_plan]
    
    scatter = ax3.scatter(comfort_vals, ml_vals, c=ml_vals, cmap='RdYlGn_r',
                         s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Correlation
    corr = np.corrcoef(comfort_vals, ml_vals)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes,
            fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Diagonal line
    ax3.plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=2, label='Perfect match')
    
    ax3.set_xlabel('Bacteria Comfort Index (%)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('ML Outbreak Probability (%)', fontweight='bold', fontsize=11)
    ax3.set_title('🤖 ML vs Comfort: Validation', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='ML Probability')
    
    # 4. Risk Agreement Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Создаём матрицу согласованности
    agreement_matrix = np.zeros((4, 4))
    risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    for zone in vaccination_plan:
        comfort_risk = zone['bacteria_comfort']['comfort_level']
        if comfort_risk == 'VERY HIGH':
            comfort_idx = 3
        elif comfort_risk == 'HIGH':
            comfort_idx = 2
        elif comfort_risk == 'MODERATE':
            comfort_idx = 1
        else:
            comfort_idx = 0
        
        ml_risk = zone.get('ml_risk', 'LOW')
        ml_idx = risk_levels.index(ml_risk) if ml_risk in risk_levels else 0
        
        agreement_matrix[comfort_idx, ml_idx] += 1
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.0f', cmap='YlGnBu',
                xticklabels=risk_levels, yticklabels=['LOW', 'MODERATE', 'HIGH', 'VERY HIGH'],
                ax=ax4, cbar_kws={'label': 'Number of Zones'})
    ax4.set_xlabel('ML Risk Level', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Comfort Risk Level', fontweight='bold', fontsize=11)
    ax4.set_title('🤖 Risk Agreement Matrix', fontsize=13, fontweight='bold')
    
    # 5. ML Risk Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ml_risks = [z.get('ml_risk', 'LOW') for z in vaccination_plan]
    risk_counts = {r: ml_risks.count(r) for r in risk_levels}
    
    colors_risk = ['#689f38', '#fbc02d', '#f57c00', '#d32f2f']
    bars = ax5.bar(risk_levels, [risk_counts[r] for r in risk_levels],
                   color=colors_risk, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, [risk_counts[r] for r in risk_levels]):
        ax5.text(bar.get_x() + bar.get_width()/2, count + 0.5,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax5.set_ylabel('Number of Zones', fontweight='bold', fontsize=11)
    ax5.set_title('🤖 ML: Risk Level Distribution', fontsize=13, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Lives Saved vs ML Probability
    ax6 = fig.add_subplot(gs[1, 2])
    lives = [z.get('lives_saved', 0) for z in vaccination_plan]
    ml_p = [z.get('ml_outbreak_probability', 0) for z in vaccination_plan]
    
    scatter2 = ax6.scatter(ml_p, lives, c=ml_p, cmap='RdYlGn_r',
                          s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    ax6.set_xlabel('ML Outbreak Probability (%)', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Lives Saved', fontweight='bold', fontsize=11)
    ax6.set_title('🤖 ML Probability vs Impact', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax6, label='ML Prob %')
    
    # 7. Comparison: Traditional vs ML
    ax7 = fig.add_subplot(gs[2, :])
    
    # Топ-10 зон по Comfort Index
    top_comfort = sorted(vaccination_plan, 
                        key=lambda x: x['bacteria_comfort']['overall_comfort'],
                        reverse=True)[:10]
    
    # Топ-10 зон по ML
    top_ml = sorted(vaccination_plan,
                   key=lambda x: x.get('ml_outbreak_probability', 0),
                   reverse=True)[:10]
    
    zone_labels = [f"Z{z['zone_number']}" for z in top_comfort]
    comfort_scores = [z['bacteria_comfort']['overall_comfort'] for z in top_comfort]
    ml_scores = [z.get('ml_outbreak_probability', 0) for z in top_comfort]
    
    x = np.arange(len(zone_labels))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, comfort_scores, width, label='Comfort Index',
                   color='#42a5f5', alpha=0.7, edgecolor='black')
    bars2 = ax7.bar(x + width/2, ml_scores, width, label='ML Probability',
                   color='#66bb6a', alpha=0.7, edgecolor='black')
    
    ax7.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax7.set_title('🤖 Top 10 Zones: Traditional Comfort Index vs ML Predictions',
                 fontsize=14, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(zone_labels, rotation=45, ha='right')
    ax7.legend(fontsize=11)
    ax7.grid(axis='y', alpha=0.3)
    ax7.set_ylim(0, 110)
    
    fig.suptitle('🤖 AI-ENHANCED EPIDEMIC PREDICTION SYSTEM\nMachine Learning Analysis',
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('ai_enhancements_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ AI visualizations saved: ai_enhancements_analysis.png")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ ИНТЕГРАЦИИ AI
# ============================================================================

def integrate_ai_components(vaccination_plan, disease, summary):
    """
    Добавляет все AI-компоненты в систему
    
    Args:
        vaccination_plan: список зон с данными
        disease: профиль заболевания
        summary: сводная статистика
    
    Returns:
        dict с AI-компонентами
    """
    
    print("\n" + "="*70)
    print("🤖 INTEGRATING AI COMPONENTS")
    print("="*70)
    
    results = {}
    
    # 1. ML-предсказание вспышек
    print("\n1️⃣ Training ML Outbreak Predictor...")
    
    predictor = OutbreakPredictor()
    training_data = predictor.create_synthetic_training_data(vaccination_plan, n_samples=500)
    ml_importances = predictor.train(training_data)
    
    # Добавляем предсказания
    vaccination_plan = add_ml_predictions_to_zones(vaccination_plan, predictor)
    
    results['ml_predictor'] = predictor
    results['ml_importances'] = ml_importances
    
    # 2. AI-ассистент
    print("\n2️⃣ Initializing AI Assistant...")
    
    assistant = AIEpidemicAssistant(vaccination_plan, disease, summary)
    results['ai_assistant'] = assistant
    
    print("✅ AI Assistant ready")
    
    # 3. Визуализации
    print("\n3️⃣ Creating AI visualizations...")
    visualize_ai_enhancements(vaccination_plan, ml_importances)
    
    # 4. Статистика
    print("\n" + "="*70)
    print("🎯 AI INTEGRATION SUMMARY")
    print("="*70)
    
    ml_high_risk = len([z for z in vaccination_plan if z.get('ml_outbreak_probability', 0) >= 50])
    ml_critical = len([z for z in vaccination_plan if z.get('ml_outbreak_probability', 0) >= 75])
    
    print(f"\n🤖 ML Predictions:")
    print(f"   Critical risk zones (≥75%): {ml_critical}")
    print(f"   High risk zones (≥50%): {ml_high_risk}")
    print(f"   Total zones analyzed: {len(vaccination_plan)}")
    
    # Сравнение с Comfort Index
    comfort_high = len([z for z in vaccination_plan 
                       if z['bacteria_comfort']['overall_comfort'] >= 60])
    
    agreement = len([z for z in vaccination_plan 
                    if (z.get('ml_outbreak_probability', 0) >= 50 and 
                        z['bacteria_comfort']['overall_comfort'] >= 60) or
                       (z.get('ml_outbreak_probability', 0) < 50 and 
                        z['bacteria_comfort']['overall_comfort'] < 60)])
    
    print(f"\n🔍 ML vs Comfort Index:")
    print(f"   Agreement: {agreement}/{len(vaccination_plan)} zones ({agreement/len(vaccination_plan)*100:.1f}%)")
    
    # Топ-3 по ML
    top_ml_zones = sorted(vaccination_plan, 
                         key=lambda x: x.get('ml_outbreak_probability', 0),
                         reverse=True)[:3]
    
    print(f"\n🏆 Top 3 Zones by ML Risk:")
    for i, zone in enumerate(top_ml_zones, 1):
        print(f"   {i}. Zone #{zone['zone_number']}: {zone.get('ml_outbreak_probability', 0):.1f}% "
              f"(Comfort: {zone['bacteria_comfort']['overall_comfort']:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ AI INTEGRATION COMPLETE!")
    print("="*70)
    
    return results, vaccination_plan


# ============================================================================
# ДЕМО ДЛЯ ХАКАТОНА
# ============================================================================

def hackathon_demo(vaccination_plan, disease, summary, ai_results):
    """
    Создаёт эффектное демо для хакатона
    """
    
    print("\n" + "="*70)
    print("🎬 HACKATHON DEMO")
    print("="*70)
    
    assistant = ai_results['ai_assistant']
    
    # 1. Показываем ML-предсказания
    print("\n1️⃣ ML PREDICTIONS:")
    print("-" * 70)
    
    critical_zones = [z for z in vaccination_plan 
                     if z.get('ml_outbreak_probability', 0) >= 75]
    
    print(f"\n⚠️ CRITICAL ALERT: {len(critical_zones)} zones with >75% outbreak probability!")
    for zone in critical_zones[:3]:
        print(f"\n   Zone #{zone['zone_number']}:")
        print(f"   - ML Probability: {zone['ml_outbreak_probability']:.1f}%")
        print(f"   - Bacteria Comfort: {zone['bacteria_comfort']['overall_comfort']:.1f}%")
        print(f"   - Lives at risk: ~{zone.get('lives_saved', 0):,}")
        print(f"   - Location: ({zone['coordinates'][0]:.4f}, {zone['coordinates'][1]:.4f})")
    
    # 2. AI Assistant demo
    print("\n\n2️⃣ AI ASSISTANT DEMO:")
    print("-" * 70)
    
    questions = [
        "Which zone needs immediate vaccination?",
        "What's the total cost to prevent this outbreak?",
        "How many lives can we save?"
    ]
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        answer = assistant.ask(q)
        print(f"🤖 AI: {answer[:200]}...")  # первые 200 символов
    
    # 3. Генерация отчёта
    print("\n\n3️⃣ GENERATING EXECUTIVE REPORT:")
    print("-" * 70)
    
    report = assistant.generate_executive_summary()
    print(report[:300] + "...")  # первые 300 символов
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE - READY FOR PRESENTATION!")
    print("="*70)


# ============================================================================
# ЭКСПОРТ AI-ДАННЫХ
# ============================================================================

def export_ai_data(vaccination_plan, filepath='ai_predictions.csv'):
    """
    Экспортирует данные с AI-предсказаниями
    """
    
    data = []
    for zone in vaccination_plan:
        data.append({
            'Zone': zone['zone_number'],
            'Priority': zone['priority'],
            'Comfort_Index': zone['bacteria_comfort']['overall_comfort'],
            'Comfort_Risk': zone['bacteria_comfort']['comfort_level'],
            'ML_Probability': zone.get('ml_outbreak_probability', 0),
            'ML_Risk': zone.get('ml_risk', 'N/A'),
            'Lives_Saved': zone.get('lives_saved', 0),
            'Cost_USD': zone['total_cost_usd'],
            'Temperature': zone['satellite_data']['temperature'],
            'Humidity': zone['satellite_data']['humidity'],
            'NDVI': zone['satellite_data']['ndvi'],
            'NDWI': zone['satellite_data']['ndwi']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('ML_Probability', ascending=False)
    df.to_csv(filepath, index=False)
    
    print(f"\n✅ AI predictions exported: {filepath}")


# ============================================================================
# ИСПОЛЬЗОВАНИЕ В NOTEBOOK
# ============================================================================

"""
# В вашем notebook, ПОСЛЕ расчёта vaccination_plan:

# Интегрировать AI
ai_results, vaccination_plan = integrate_ai_components(vaccination_plan, disease, summary)

# Экспортировать данные
export_ai_data(vaccination_plan, 'ai_predictions.csv')

# Запустить демо (для хакатона)
hackathon_demo(vaccination_plan, disease, summary, ai_results)

# Использовать AI-ассистента
assistant = ai_results['ai_assistant']
answer = assistant.ask("What should we do first?")
print(answer)

# Сохранить ML-модель
ai_results['ml_predictor'].save('outbreak_ml_model.pkl')

print("\\n🎉 AI components fully integrated!")
"""
