# ============================================================================
# UNIFIED VISUALIZATION: Satellite Data + Bacteria Comfort + Lives Saved
# ============================================================================
# Вставьте этот код ПОСЛЕ расчёта vaccination_plan в ваш notebook
# Создаёт единую интерактивную карту со всей информацией
# ============================================================================

import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import branca.colormap as cm

print("="*70)
print("🗺️ СОЗДАНИЕ ОБЪЕДИНЁННОЙ ИНТЕРАКТИВНОЙ КАРТЫ")
print("="*70)

# --------------------------------------------------------------------------
# ФУНКЦИЯ: Расчёт индекса комфортности бактерий (Bacteria Comfort Index)
# --------------------------------------------------------------------------

def calculate_bacteria_comfort_index(conditions: dict, disease_profile: dict) -> dict:
    """
    Рассчитывает детальный индекс комфортности среды для патогена
    
    Args:
        conditions: спутниковые и климатические данные
        disease_profile: профиль заболевания
    
    Returns:
        dict с подробными оценками комфортности
    """
    
    scores = {}
    
    # 1. ТЕМПЕРАТУРНАЯ КОМФОРТНОСТЬ (0-100%)
    temp = conditions['temperature']
    temp_min, temp_max = disease_profile['temp_range']
    temp_opt = disease_profile['temp_optimal']
    
    if temp_min <= temp <= temp_max:
        # Чем ближе к оптимуму, тем выше оценка
        temp_deviation = abs(temp - temp_opt)
        temp_range = temp_max - temp_min
        temp_comfort = max(0, 100 * (1 - temp_deviation / temp_range))
    else:
        temp_comfort = 0
    
    scores['temperature_comfort'] = temp_comfort
    
    # 2. ВЛАЖНОСТНАЯ КОМФОРТНОСТЬ (0-100%)
    humidity = conditions.get('humidity', 50)
    hum_min, hum_max = disease_profile.get('humidity_range', (0, 100))
    
    if hum_min <= humidity <= hum_max:
        # Оптимум в середине диапазона
        hum_opt = (hum_min + hum_max) / 2
        hum_deviation = abs(humidity - hum_opt)
        hum_range = hum_max - hum_min
        humidity_comfort = max(0, 100 * (1 - hum_deviation / hum_range))
    else:
        humidity_comfort = 0
    
    scores['humidity_comfort'] = humidity_comfort
    
    # 3. ВОДНАЯ СРЕДА (0-100%)
    if disease_profile.get('water_required', False):
        ndwi = conditions.get('ndwi', 0)
        # NDWI от -1 до 1, где >0.1 = вода
        if ndwi > 0.3:
            water_comfort = 100
        elif ndwi > 0.1:
            water_comfort = 70
        elif ndwi > 0:
            water_comfort = 40
        else:
            water_comfort = 10
    else:
        # Для болезней не требующих воду
        water_comfort = 50  # нейтрально
    
    scores['water_comfort'] = water_comfort
    
    # 4. ВЕГЕТАЦИОННАЯ СРЕДА (0-100%)
    ndvi = conditions.get('ndvi', 0)
    ndvi_pref = disease_profile.get('ndvi_preference', 'medium')
    
    if ndvi_pref == 'high':
        # Высокая растительность предпочтительна
        vegetation_comfort = min(100, max(0, ndvi * 100))
    elif ndvi_pref == 'low':
        # Низкая растительность предпочтительна
        vegetation_comfort = min(100, max(0, (1 - ndvi) * 100))
    else:  # medium
        # Средняя растительность оптимальна
        vegetation_comfort = 100 * (1 - abs(ndvi - 0.5) / 0.5)
    
    scores['vegetation_comfort'] = vegetation_comfort
    
    # 5. ОБЩИЙ ИНДЕКС КОМФОРТНОСТИ (взвешенная сумма)
    weights = {
        'temperature_comfort': 0.35,   # температура - главный фактор
        'humidity_comfort': 0.25,      # влажность - важна
        'water_comfort': 0.25,         # вода - для leptospirosis критична
        'vegetation_comfort': 0.15     # растительность - дополнительный фактор
    }
    
    overall_comfort = sum(scores[k] * weights[k] for k in weights.keys())
    scores['overall_comfort'] = overall_comfort
    
    # 6. ТЕКСТОВАЯ ОЦЕНКА
    if overall_comfort >= 80:
        comfort_level = "VERY HIGH"
        comfort_color = "#d32f2f"  # красный
        risk_level = "CRITICAL"
    elif overall_comfort >= 60:
        comfort_level = "HIGH"
        comfort_color = "#f57c00"  # оранжевый
        risk_level = "HIGH"
    elif overall_comfort >= 40:
        comfort_level = "MODERATE"
        comfort_color = "#fbc02d"  # жёлтый
        risk_level = "MEDIUM"
    elif overall_comfort >= 20:
        comfort_level = "LOW"
        comfort_color = "#689f38"  # зелёный
        risk_level = "LOW"
    else:
        comfort_level = "VERY LOW"
        comfort_color = "#388e3c"  # тёмно-зелёный
        risk_level = "MINIMAL"
    
    scores['comfort_level'] = comfort_level
    scores['comfort_color'] = comfort_color
    scores['risk_level'] = risk_level
    
    return scores

# --------------------------------------------------------------------------
# ФУНКЦИЯ: Создание объединённой карты
# --------------------------------------------------------------------------

def create_unified_map(vaccination_plan, epicenter, disease_profile, analysis):
    """
    Создаёт объединённую интерактивную карту с:
    - Спутниковыми данными
    - Индексом комфортности бактерий
    - Спасёнными жизнями
    - Приоритетами вакцинации
    """
    
    print("\n⏳ Обогащение данных...")
    
    # Обогащаем каждую зону информацией
    for zone in vaccination_plan:
        # 1. Спасённые жизни
        prevented = zone['predicted_cases_without_vaccine'] - zone['predicted_cases_with_vaccine']
        zone['prevented_cases_with_vaccine'] = prevented
        zone['lives_saved'] = int(prevented * disease_profile['fatality_rate'])
        
        # 2. Индекс комфортности бактерий
        conditions = zone.get('conditions', {})
        if not conditions:
            # Если данных нет, получаем их
            conditions = get_satellite_conditions(
                zone['coordinates'][0], 
                zone['coordinates'][1], 
                disease_profile
            )
            zone['conditions'] = conditions
        
        comfort_scores = calculate_bacteria_comfort_index(conditions, disease_profile)
        zone['bacteria_comfort'] = comfort_scores
        
        # 3. Детальные спутниковые данные
        zone['satellite_data'] = {
            'temperature': conditions.get('temperature', 'N/A'),
            'humidity': conditions.get('humidity', 'N/A'),
            'ndvi': conditions.get('ndvi', 'N/A'),
            'ndwi': conditions.get('ndwi', 'N/A'),
            'water_present': conditions.get('water_presence', False),
            'precipitation_30d': conditions.get('precipitation_30d', 'N/A'),
            'sources': conditions.get('data_sources', {})
        }
    
    print("✅ Данные обогащены")
    print("\n⏳ Создание карты...")
    
    # Создаём базовую карту
    m = folium.Map(
        location=[epicenter['lat'], epicenter['lon']],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Добавляем альтернативные базовые слои
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    # Маркер эпицентра
    folium.Marker(
        [epicenter['lat'], epicenter['lon']],
        popup=f"<b>EPICENTER</b><br>{epicenter.get('location', 'Outbreak Location')}",
        icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
        tooltip="Outbreak Epicenter"
    ).add_to(m)
    
    # ---------- СЛОЙ 1: CIRCLES - Спасённые жизни и приоритеты ----------
    
    lives_layer = folium.FeatureGroup(name='💊 Lives Saved & Priorities', show=True)
    
    priority_colors = {
        'CRITICAL': '#d32f2f',
        'HIGH': '#f57c00',
        'MEDIUM': '#fbc02d',
        'LOW': '#689f38'
    }
    
    for zone in vaccination_plan:
        lat, lon = zone['coordinates']
        lives = zone['lives_saved']
        
        popup_html = f"""
        <div style="font-family: Arial; width: 350px; max-height: 500px; overflow-y: auto;">
            <h3 style="color: {priority_colors[zone['priority']]}; margin: 0 0 10px 0; 
                       border-bottom: 2px solid {priority_colors[zone['priority']]}; padding-bottom: 5px;">
                🗺️ Zone #{zone['zone_number']} - {zone['priority']}
            </h3>
            
            <!-- LIVES SAVED -->
            <div style="background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%); 
                        color: white; padding: 12px; border-radius: 8px; margin-bottom: 10px; 
                        text-align: center;">
                <div style="font-size: 28px; font-weight: bold;">✅ {lives:,}</div>
                <div style="font-size: 14px; margin-top: 5px;">ЖИЗНЕЙ СПАСЕНО</div>
            </div>
            
            <!-- BACTERIA COMFORT INDEX -->
            <div style="background: linear-gradient(135deg, {zone['bacteria_comfort']['comfort_color']} 0%, 
                        {zone['bacteria_comfort']['comfort_color']}aa 100%); 
                        color: white; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                <div style="font-size: 11px; opacity: 0.9;">BACTERIA COMFORT INDEX</div>
                <div style="font-size: 20px; font-weight: bold;">
                    {zone['bacteria_comfort']['overall_comfort']:.1f}%
                </div>
                <div style="font-size: 12px; margin-top: 3px;">
                    {zone['bacteria_comfort']['comfort_level']} RISK
                </div>
            </div>
            
            <!-- DETAILED COMFORT SCORES -->
            <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: 12px;">
                    🦠 Environmental Factors:
                </div>
                <div style="font-size: 11px;">
                    <div style="margin: 3px 0; display: flex; justify-content: space-between;">
                        <span>🌡️ Temperature:</span>
                        <span style="font-weight: bold;">{zone['bacteria_comfort']['temperature_comfort']:.0f}%</span>
                    </div>
                    <div style="margin: 3px 0; display: flex; justify-content: space-between;">
                        <span>💧 Humidity:</span>
                        <span style="font-weight: bold;">{zone['bacteria_comfort']['humidity_comfort']:.0f}%</span>
                    </div>
                    <div style="margin: 3px 0; display: flex; justify-content: space-between;">
                        <span>🌊 Water:</span>
                        <span style="font-weight: bold;">{zone['bacteria_comfort']['water_comfort']:.0f}%</span>
                    </div>
                    <div style="margin: 3px 0; display: flex; justify-content: space-between;">
                        <span>🌿 Vegetation:</span>
                        <span style="font-weight: bold;">{zone['bacteria_comfort']['vegetation_comfort']:.0f}%</span>
                    </div>
                </div>
            </div>
            
            <!-- SATELLITE DATA -->
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: 12px;">
                    🛰️ Satellite & Climate Data:
                </div>
                <table style="width: 100%; font-size: 10px;">
                    <tr>
                        <td>Temperature:</td>
                        <td><b>{zone['satellite_data']['temperature']:.1f}°C</b></td>
                    </tr>
                    <tr>
                        <td>Humidity:</td>
                        <td><b>{zone['satellite_data']['humidity']:.0f}%</b></td>
                    </tr>
                    <tr>
                        <td>NDVI (vegetation):</td>
                        <td><b>{zone['satellite_data']['ndvi']:.3f}</b></td>
                    </tr>
                    <tr>
                        <td>NDWI (water):</td>
                        <td><b>{zone['satellite_data']['ndwi']:.3f}</b></td>
                    </tr>
                    <tr>
                        <td>Precipitation (30d):</td>
                        <td><b>{zone['satellite_data']['precipitation_30d']:.0f} mm</b></td>
                    </tr>
                    <tr>
                        <td>Water present:</td>
                        <td><b>{'Yes ✓' if zone['satellite_data']['water_present'] else 'No ✗'}</b></td>
                    </tr>
                </table>
                <div style="font-size: 9px; margin-top: 5px; opacity: 0.7;">
                    Source: {zone['satellite_data']['sources'].get('satellite', 'N/A')}
                </div>
            </div>
            
            <!-- VACCINATION IMPACT -->
            <div style="background-color: #fff3e0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: 12px;">
                    💉 Vaccination Impact:
                </div>
                <table style="width: 100%; font-size: 11px;">
                    <tr>
                        <td>Population:</td>
                        <td><b>{zone['population']:,}</b></td>
                    </tr>
                    <tr>
                        <td>Target vaccination:</td>
                        <td><b>{zone['target_population']:,} ({zone['coverage_percent']:.1f}%)</b></td>
                    </tr>
                    <tr>
                        <td>Total doses:</td>
                        <td><b>{zone['total_doses']:,}</b></td>
                    </tr>
                    <tr style="background-color: #ffebee;">
                        <td>Without vaccine:</td>
                        <td><b style="color: #c62828;">{zone['predicted_cases_without_vaccine']:,} cases</b></td>
                    </tr>
                    <tr style="background-color: #e8f5e9;">
                        <td>With vaccine:</td>
                        <td><b style="color: #2e7d32;">{zone['predicted_cases_with_vaccine']:,} cases</b></td>
                    </tr>
                    <tr>
                        <td>Prevented cases:</td>
                        <td><b style="color: #1565c0;">{zone['prevented_cases_with_vaccine']:,}</b></td>
                    </tr>
                </table>
            </div>
            
            <!-- FINANCIAL -->
            <div style="background-color: #f3e5f5; padding: 10px; border-radius: 5px;">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: 12px;">
                    💰 Financial:
                </div>
                <div style="font-size: 11px;">
                    <div style="margin: 3px 0; display: flex; justify-content: space-between;">
                        <span>Total cost:</span>
                        <span style="font-weight: bold;">${zone['total_cost_usd']:,.0f}</span>
                    </div>
                    <div style="margin: 3px 0; display: flex; justify-content: space-between;">
                        <span>Cost per life:</span>
                        <span style="font-weight: bold; color: #1565c0;">
                            ${zone['total_cost_usd']/lives if lives > 0 else 0:,.0f}
                        </span>
                    </div>
                    <div style="margin: 3px 0; display: flex; justify-content: space-between;">
                        <span>Teams needed:</span>
                        <span style="font-weight: bold;">{zone['teams_needed']}</span>
                    </div>
                    <div style="margin: 3px 0; display: flex; justify-content: space-between;">
                        <span>Timeline:</span>
                        <span style="font-weight: bold;">{zone['completion_time']}</span>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 10px; text-align: center; font-size: 10px; color: #666;">
                📍 Coordinates: {lat:.4f}, {lon:.4f}
            </div>
        </div>
        """
        
        # Размер круга зависит от спасённых жизней
        radius = max(3000, lives * 100)
        
        folium.Circle(
            location=[lat, lon],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=370),
            tooltip=f"Zone {zone['zone_number']}: {lives:,} lives | {zone['bacteria_comfort']['overall_comfort']:.0f}% comfort",
            color=priority_colors[zone['priority']],
            fill=True,
            fillColor=priority_colors[zone['priority']],
            fillOpacity=0.4,
            weight=2
        ).add_to(lives_layer)
    
    lives_layer.add_to(m)
    
    # ---------- СЛОЙ 2: HEATMAP - Индекс комфортности бактерий ----------
    
    comfort_heat_data = []
    for zone in vaccination_plan:
        lat, lon = zone['coordinates']
        comfort = zone['bacteria_comfort']['overall_comfort']
        if comfort > 0:
            # Интенсивность пропорциональна комфортности
            comfort_heat_data.append([lat, lon, comfort / 100])
    
    if comfort_heat_data:
        comfort_heatmap = plugins.HeatMap(
            comfort_heat_data,
            name='🦠 Bacteria Comfort Heatmap',
            min_opacity=0.3,
            max_zoom=13,
            radius=20,
            blur=25,
            gradient={
                0.0: '#00ff00',  # низкий комфорт = зелёный
                0.4: '#ffff00',  # средний = жёлтый
                0.7: '#ff8800',  # высокий = оранжевый
                1.0: '#ff0000'   # очень высокий = красный
            },
            show=False  # по умолчанию выключен
        )
        m.add_child(comfort_heatmap)
    
    # ---------- СЛОЙ 3: HEATMAP - Спасённые жизни ----------
    
    lives_heat_data = []
    for zone in vaccination_plan:
        lat, lon = zone['coordinates']
        lives = zone['lives_saved']
        if lives > 0:
            lives_heat_data.append([lat, lon, lives / 100])
    
    if lives_heat_data:
        lives_heatmap = plugins.HeatMap(
            lives_heat_data,
            name='✅ Lives Saved Heatmap',
            min_opacity=0.2,
            max_zoom=13,
            radius=25,
            blur=35,
            gradient={
                0.0: '#4caf50',  # мало = зелёный
                0.5: '#ffc107',  # средне = золотой
                1.0: '#2196f3'   # много = синий (позитивный цвет)
            },
            show=False
        )
        m.add_child(lives_heatmap)
    
    # ---------- СЛОЙ 4: Маркеры с данными спутников ----------
    
    satellite_layer = folium.FeatureGroup(name='🛰️ Satellite Data Points', show=False)
    
    for zone in vaccination_plan:
        lat, lon = zone['coordinates']
        sat_data = zone['satellite_data']
        
        # Иконка зависит от наличия воды
        icon_name = 'tint' if sat_data['water_present'] else 'leaf'
        icon_color = 'blue' if sat_data['water_present'] else 'green'
        
        marker_popup = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 0 0 10px 0;">🛰️ Satellite Data</h4>
            <table style="width: 100%; font-size: 11px;">
                <tr><td>NDVI:</td><td><b>{sat_data['ndvi']:.3f}</b></td></tr>
                <tr><td>NDWI:</td><td><b>{sat_data['ndwi']:.3f}</b></td></tr>
                <tr><td>Temperature:</td><td><b>{sat_data['temperature']:.1f}°C</b></td></tr>
                <tr><td>Humidity:</td><td><b>{sat_data['humidity']:.0f}%</b></td></tr>
            </table>
            <div style="margin-top: 8px; font-size: 9px;">
                Source: {sat_data['sources'].get('satellite', 'N/A')}
            </div>
        </div>
        """
        
        folium.Marker(
            [lat, lon],
            popup=marker_popup,
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa'),
            tooltip=f"NDVI: {sat_data['ndvi']:.2f}, NDWI: {sat_data['ndwi']:.2f}"
        ).add_to(satellite_layer)
    
    satellite_layer.add_to(m)
    
    # ---------- ЛЕГЕНДА ----------
    
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 280px; 
                background-color: white; border: 3px solid grey; z-index: 9999; 
                padding: 15px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.3);
                font-family: Arial;">
        
        <h3 style="margin: 0 0 12px 0; font-size: 16px; border-bottom: 2px solid #333; padding-bottom: 5px;">
            🗺️ {disease_profile['name']}
        </h3>
        
        <div style="margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 13px; margin-bottom: 5px;">
                Priority Levels:
            </div>
            <div style="font-size: 11px;">
                <span style="color: #d32f2f; font-size: 14px;">⬤</span> CRITICAL<br>
                <span style="color: #f57c00; font-size: 14px;">⬤</span> HIGH<br>
                <span style="color: #fbc02d; font-size: 14px;">⬤</span> MEDIUM<br>
                <span style="color: #689f38; font-size: 14px;">⬤</span> LOW
            </div>
        </div>
        
        <div style="margin-bottom: 12px; padding: 8px; background-color: #f0f0f0; border-radius: 5px;">
            <div style="font-weight: bold; font-size: 12px; margin-bottom: 5px;">
                🦠 Bacteria Comfort Index:
            </div>
            <div style="font-size: 10px;">
                Measures environmental<br>
                favorability for pathogen:<br>
                <span style="color: #d32f2f;">■</span> 80-100%: Very High Risk<br>
                <span style="color: #f57c00;">■</span> 60-80%: High Risk<br>
                <span style="color: #fbc02d;">■</span> 40-60%: Moderate Risk<br>
                <span style="color: #689f38;">■</span> <40%: Low Risk
            </div>
        </div>
        
        <div style="margin-bottom: 10px;">
            <div style="font-weight: bold; font-size: 12px; margin-bottom: 5px;">
                Map Features:
            </div>
            <div style="font-size: 10px;">
                <b>Circle size:</b> Lives saved<br>
                <b>Circle color:</b> Priority level<br>
                <b>Click zone:</b> Full details<br>
                <b>Heatmaps:</b> Toggle layers
            </div>
        </div>
        
        <div style="font-size: 9px; padding-top: 8px; border-top: 1px solid #ccc; color: #666;">
            Data sources:<br>
            🛰️ Sentinel-2 / Landsat<br>
            🌡️ NASA POWER / Open-Meteo
        </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Добавляем контроль слоёв
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)
    
    # Добавляем мини-карту
    minimap = plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)
    
    # Добавляем полноэкранный режим
    plugins.Fullscreen(
        position='topleft',
        title='Full Screen',
        title_cancel='Exit Full Screen',
        force_separate_button=True
    ).add_to(m)
    
    # Добавляем измеритель расстояний
    plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    return m

# --------------------------------------------------------------------------
# ОСНОВНОЙ КОД ЗАПУСКА
# --------------------------------------------------------------------------

print("\n⏳ Обогащение данных спасёнными жизнями...")
for zone in vaccination_plan:
    prevented = zone['predicted_cases_without_vaccine'] - zone['predicted_cases_with_vaccine']
    zone['prevented_cases_with_vaccine'] = prevented
    zone['lives_saved'] = int(prevented * disease['fatality_rate'])

total_lives_saved = sum(z['lives_saved'] for z in vaccination_plan)
print(f"✅ Всего спасено жизней: {total_lives_saved:,}")

# Создаём объединённую карту
print("\n⏳ Создание объединённой интерактивной карты...")
unified_map = create_unified_map(vaccination_plan, analysis['epicenter'], disease, analysis)
unified_map.save('unified_satellite_lives_map.html')
print("✅ Карта сохранена: unified_satellite_lives_map.html")

# --------------------------------------------------------------------------
# ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: Графики комфортности
# --------------------------------------------------------------------------

print("\n⏳ Создание аналитических графиков...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Карта комфортности (scatter plot)
ax1 = fig.add_subplot(gs[0, :2])

lats = [z['coordinates'][0] for z in vaccination_plan]
lons = [z['coordinates'][1] for z in vaccination_plan]
comforts = [z['bacteria_comfort']['overall_comfort'] for z in vaccination_plan]
lives_list = [z['lives_saved'] for z in vaccination_plan]

scatter = ax1.scatter(lons, lats, c=comforts, s=[l*2 for l in lives_list],  
                     cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=1)
ax1.scatter([analysis['epicenter']['lon']], [analysis['epicenter']['lat']], 
           marker='*', s=500, c='red', edgecolors='black', linewidth=2, zorder=5)

cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Bacteria Comfort Index (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax1.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax1.set_title('Geographic Distribution: Bacteria Comfort vs Lives Saved', 
             fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Детализация факторов комфортности
ax2 = fig.add_subplot(gs[0, 2])

avg_comfort = {
    'Temperature': np.mean([z['bacteria_comfort']['temperature_comfort'] for z in vaccination_plan]),
    'Humidity': np.mean([z['bacteria_comfort']['humidity_comfort'] for z in vaccination_plan]),
    'Water': np.mean([z['bacteria_comfort']['water_comfort'] for z in vaccination_plan]),
    'Vegetation': np.mean([z['bacteria_comfort']['vegetation_comfort'] for z in vaccination_plan])
}

factors = list(avg_comfort.keys())
values = list(avg_comfort.values())
colors_factors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

bars = ax2.barh(factors, values, color=colors_factors, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, values):
    ax2.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
            ha='left', va='center', fontsize=11, fontweight='bold')

ax2.set_xlabel('Average Comfort (%)', fontsize=11, fontweight='bold')
ax2.set_title('Environmental Factors\n(Average across all zones)', 
             fontsize=12, fontweight='bold')
ax2.set_xlim(0, 110)
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# 3. Корреляция: Комфортность vs Спасённые жизни
ax3 = fig.add_subplot(gs[1, 0])

ax3.scatter(comforts, lives_list, c=comforts, cmap='RdYlGn_r', 
           s=100, alpha=0.6, edgecolors='black')

# Линия тренда
z = np.polyfit(comforts, lives_list, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(comforts), max(comforts), 100)
ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')

ax3.set_xlabel('Bacteria Comfort Index (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Lives Saved', fontsize=11, fontweight='bold')
ax3.set_title('Correlation: Comfort vs Impact', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Добавляем корреляцию
corr = np.corrcoef(comforts, lives_list)[0, 1]
ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Распределение зон по уровням комфортности
ax4 = fig.add_subplot(gs[1, 1])

comfort_levels = {}
for zone in vaccination_plan:
    level = zone['bacteria_comfort']['comfort_level']
    if level not in comfort_levels:
        comfort_levels[level] = 0
    comfort_levels[level] += 1

levels = list(comfort_levels.keys())
counts = list(comfort_levels.values())

level_colors = {
    'VERY HIGH': '#d32f2f',
    'HIGH': '#f57c00',
    'MODERATE': '#fbc02d',
    'LOW': '#689f38',
    'VERY LOW': '#388e3c'
}
pie_colors = [level_colors.get(l, '#999999') for l in levels]

wedges, texts, autotexts = ax4.pie(counts, labels=levels, autopct='%1.1f%%',
                                    colors=pie_colors, startangle=90,
                                    textprops={'fontweight': 'bold', 'fontsize': 10})

ax4.set_title('Zones by Comfort Level', fontsize=12, fontweight='bold')

# 5. Top 10 зон по комфортности
ax5 = fig.add_subplot(gs[1, 2])

sorted_by_comfort = sorted(vaccination_plan, 
                           key=lambda x: x['bacteria_comfort']['overall_comfort'], 
                           reverse=True)[:10]

zone_labels_comfort = [f"Z{z['zone_number']}" for z in sorted_by_comfort]
comfort_values = [z['bacteria_comfort']['overall_comfort'] for z in sorted_by_comfort]
comfort_colors_bars = [z['bacteria_comfort']['comfort_color'] for z in sorted_by_comfort]

bars = ax5.barh(zone_labels_comfort, comfort_values, color=comfort_colors_bars, 
               alpha=0.7, edgecolor='black')
for bar, val in zip(bars, comfort_values):
    ax5.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
            ha='left', va='center', fontsize=9, fontweight='bold')

ax5.set_xlabel('Comfort Index (%)', fontsize=11, fontweight='bold')
ax5.set_title('Top 10 Highest Risk Zones\n(by Bacteria Comfort)', 
             fontsize=12, fontweight='bold')
ax5.set_xlim(0, 110)
ax5.grid(axis='x', alpha=0.3)
ax5.invert_yaxis()

# 6. Спутниковые индексы (NDVI vs NDWI)
ax6 = fig.add_subplot(gs[2, 0])

ndvi_values = [z['satellite_data']['ndvi'] for z in vaccination_plan if isinstance(z['satellite_data']['ndvi'], (int, float))]
ndwi_values = [z['satellite_data']['ndwi'] for z in vaccination_plan if isinstance(z['satellite_data']['ndwi'], (int, float))]
comfort_for_scatter = [z['bacteria_comfort']['overall_comfort'] for z in vaccination_plan 
                       if isinstance(z['satellite_data']['ndvi'], (int, float))]

scatter = ax6.scatter(ndvi_values, ndwi_values, c=comfort_for_scatter, 
                     s=100, cmap='RdYlGn_r', alpha=0.7, edgecolors='black')

ax6.axhline(y=0.1, color='blue', linestyle='--', alpha=0.5, label='Water threshold')
ax6.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='Vegetation threshold')

ax6.set_xlabel('NDVI (Vegetation Index)', fontsize=11, fontweight='bold')
ax6.set_ylabel('NDWI (Water Index)', fontsize=11, fontweight='bold')
ax6.set_title('Satellite Indices Distribution', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Comfort %', fontsize=9)

# 7. Сводная таблица
ax7 = fig.add_subplot(gs[2, 1:])
ax7.axis('off')

# Статистика
avg_overall_comfort = np.mean([z['bacteria_comfort']['overall_comfort'] for z in vaccination_plan])
max_comfort_zone = max(vaccination_plan, key=lambda x: x['bacteria_comfort']['overall_comfort'])
min_comfort_zone = min(vaccination_plan, key=lambda x: x['bacteria_comfort']['overall_comfort'])

zones_high_risk = len([z for z in vaccination_plan if z['bacteria_comfort']['overall_comfort'] >= 60])
zones_low_risk = len([z for z in vaccination_plan if z['bacteria_comfort']['overall_comfort'] < 40])

avg_temp = np.mean([z['satellite_data']['temperature'] for z in vaccination_plan 
                    if isinstance(z['satellite_data']['temperature'], (int, float))])
avg_humidity = np.mean([z['satellite_data']['humidity'] for z in vaccination_plan 
                       if isinstance(z['satellite_data']['humidity'], (int, float))])
avg_ndvi = np.mean([z['satellite_data']['ndvi'] for z in vaccination_plan 
                   if isinstance(z['satellite_data']['ndvi'], (int, float))])
avg_ndwi = np.mean([z['satellite_data']['ndwi'] for z in vaccination_plan 
                   if isinstance(z['satellite_data']['ndwi'], (int, float))])

summary_text = f"""
╔═══════════════════════════════════════════════════════════════════╗
║  COMPREHENSIVE ANALYSIS SUMMARY - {disease['name']}  
╚═══════════════════════════════════════════════════════════════════╝

🦠 BACTERIA COMFORT INDEX ANALYSIS:

Average Comfort Across All Zones: {avg_overall_comfort:.1f}%

Highest Risk Zone:
  • Zone #{max_comfort_zone['zone_number']}
  • Comfort Index: {max_comfort_zone['bacteria_comfort']['overall_comfort']:.1f}%
  • Risk Level: {max_comfort_zone['bacteria_comfort']['comfort_level']}
  • Lives Saved: {max_comfort_zone['lives_saved']:,}

Lowest Risk Zone:
  • Zone #{min_comfort_zone['zone_number']}
  • Comfort Index: {min_comfort_zone['bacteria_comfort']['overall_comfort']:.1f}%
  • Risk Level: {min_comfort_zone['bacteria_comfort']['comfort_level']}
  • Lives Saved: {min_comfort_zone['lives_saved']:,}

Distribution:
  • High/Very High Risk Zones (≥60%): {zones_high_risk}
  • Low/Very Low Risk Zones (<40%): {zones_low_risk}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛰️ SATELLITE & CLIMATE DATA (Average):

Temperature: {avg_temp:.1f}°C
Humidity: {avg_humidity:.0f}%
NDVI (Vegetation): {avg_ndvi:.3f}
NDWI (Water): {avg_ndwi:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💉 VACCINATION IMPACT:

Total Lives Saved: {total_lives_saved:,}
Total Cases Prevented: {summary['prevented_cases']:,}
Total Cost: ${summary['total_cost_usd']:,.0f}
Cost per Life: ${summary['cost_per_prevented_death']:,.0f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 KEY INSIGHT:

Correlation between Bacteria Comfort and Lives Saved: {corr:.3f}

This {'positive' if corr > 0 else 'negative'} correlation indicates that zones with
{'higher' if corr > 0 else 'lower'} environmental suitability for the pathogen tend to have
{'more' if corr > 0 else 'fewer'} lives saved through vaccination interventions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

fig.suptitle(f'UNIFIED ANALYSIS: Satellite Data + Bacteria Comfort + Vaccination Impact\n{disease["name"]}',
            fontsize=18, fontweight='bold', y=0.98)

plt.savefig('unified_analysis_charts.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Графики сохранены: unified_analysis_charts.png")

# --------------------------------------------------------------------------
# ЭКСПОРТ ДАННЫХ
# --------------------------------------------------------------------------

print("\n⏳ Экспорт данных в CSV...")

export_data = []
for zone in vaccination_plan:
    export_data.append({
        'Zone': zone['zone_number'],
        'Priority': zone['priority'],
        'Latitude': zone['coordinates'][0],
        'Longitude': zone['coordinates'][1],
        'Population': zone['population'],
        'Lives_Saved': zone['lives_saved'],
        'Prevented_Cases': zone['prevented_cases_with_vaccine'],
        'Cost_USD': zone['total_cost_usd'],
        'Cost_Per_Life': zone['total_cost_usd'] / zone['lives_saved'] if zone['lives_saved'] > 0 else 0,
        
        # Bacteria Comfort
        'Comfort_Overall': zone['bacteria_comfort']['overall_comfort'],
        'Comfort_Level': zone['bacteria_comfort']['comfort_level'],
        'Comfort_Temperature': zone['bacteria_comfort']['temperature_comfort'],
        'Comfort_Humidity': zone['bacteria_comfort']['humidity_comfort'],
        'Comfort_Water': zone['bacteria_comfort']['water_comfort'],
        'Comfort_Vegetation': zone['bacteria_comfort']['vegetation_comfort'],
        
        # Satellite Data
        'Temperature_C': zone['satellite_data']['temperature'],
        'Humidity_Percent': zone['satellite_data']['humidity'],
        'NDVI': zone['satellite_data']['ndvi'],
        'NDWI': zone['satellite_data']['ndwi'],
        'Water_Present': zone['satellite_data']['water_present'],
        'Precipitation_30d_mm': zone['satellite_data']['precipitation_30d'],
        'Data_Source': zone['satellite_data']['sources'].get('satellite', 'N/A')
    })

df_unified = pd.DataFrame(export_data)
df_unified = df_unified.sort_values('Lives_Saved', ascending=False)
df_unified.to_csv('unified_satellite_comfort_lives_data.csv', index=False)
print("✅ Данные сохранены: unified_satellite_comfort_lives_data.csv")

# --------------------------------------------------------------------------
# ИТОГОВАЯ СТАТИСТИКА
# --------------------------------------------------------------------------

print("\n" + "="*70)
print("📊 ИТОГОВАЯ СТАТИСТИКА")
print("="*70)

print(f"\n✅ ВСЕГО СПАСЕНО ЖИЗНЕЙ: {total_lives_saved:,}")
print(f"   Предотвращено случаев: {summary['prevented_cases']:,}")
print(f"   Общая стоимость: ${summary['total_cost_usd']:,.2f}")

print(f"\n🦠 BACTERIA COMFORT INDEX:")
print(f"   Средний индекс комфортности: {avg_overall_comfort:.1f}%")
print(f"   Высокий риск (≥60%): {zones_high_risk} зон")
print(f"   Низкий риск (<40%): {zones_low_risk} зон")

print(f"\n🛰️ SATELLITE DATA (средние):")
print(f"   Температура: {avg_temp:.1f}°C")
print(f"   Влажность: {avg_humidity:.0f}%")
print(f"   NDVI: {avg_ndvi:.3f}")
print(f"   NDWI: {avg_ndwi:.3f}")

print(f"\n🏆 TOP 5 ЗОН (по спасённым жизням):")
top5_lives = sorted(vaccination_plan, key=lambda x: x['lives_saved'], reverse=True)[:5]
for z in top5_lives:
    print(f"   Zone #{z['zone_number']}: {z['lives_saved']:,} жизней, "
          f"Comfort {z['bacteria_comfort']['overall_comfort']:.0f}%, "
          f"{z['priority']}")

print("\n" + "="*70)
print("✅ ВСЕ ФАЙЛЫ СОЗДАНЫ!")
print("="*70)
print("\nФайлы:")
print("  1. unified_satellite_lives_map.html - интерактивная карта")
print("  2. unified_analysis_charts.png - аналитические графики")
print("  3. unified_satellite_comfort_lives_data.csv - полная таблица данных")

print("\nФункции карты:")
print("  • Переключайте слои в левом верхнем углу")
print("  • Кликайте на зоны для полной информации")
print("  • Используйте измеритель расстояний")
print("  • Переключайте базовые карты")
print("  • Используйте полноэкранный режим")

# Скачивание для Colab
try:
    from google.colab import files
    files.download('unified_satellite_lives_map.html')
    files.download('unified_analysis_charts.png')
    files.download('unified_satellite_comfort_lives_data.csv')
    print("\n✅ Файлы готовы к скачиванию!")
except:
    print("\n💡 Файлы сохранены в директории notebook")

print("\n" + "="*70)
