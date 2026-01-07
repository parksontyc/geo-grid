import json

# 讀取 notebook
notebook_path = r'c:\labs\geo-grid\src\main_analyzed.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 新增 markdown cell
new_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 6.4 多商業區漸層圖（顯示所有 Clusters）"]
}

# 新增 code cell
new_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 多商業區漸層圖（獨立大圖）\n",
        "fig, ax = plt.subplots(1, 1, figsize=(20, 16))\n",
        "\n",
        "# 底層：所有網格淡色背景\n",
        "grid_with_kde.plot(ax=ax, color='whitesmoke', edgecolor='white', linewidth=0.1, alpha=0.5)\n",
        "\n",
        "# 核心層：HH格子依 Score_i 顯示藍→綠→黃漸層\n",
        "hh_grids = grid_with_kde[grid_with_kde['HH_flag']]\n",
        "\n",
        "if len(hh_grids) > 0:\n",
        "    # 使用 viridis_r 色階：深藍(高分) -> 綠 -> 黃(低分)\n",
        "    hh_grids.plot(column='Score_i', cmap='viridis_r', ax=ax, \n",
        "                  edgecolor='black', linewidth=0.5, alpha=0.9,\n",
        "                  legend=True, \n",
        "                  legend_kwds={\n",
        "                      'label': 'HH核心區 餐飲火力指數 (Score_i)', \n",
        "                      'shrink': 0.8,\n",
        "                      'orientation': 'horizontal',\n",
        "                      'pad': 0.05\n",
        "                  },\n",
        "                  vmin=hh_grids['Score_i'].quantile(0.1),\n",
        "                  vmax=hh_grids['Score_i'].max())\n",
        "    \n",
        "    print(f\"HH核心區統計:\")\n",
        "    print(f\"  總格子數: {len(hh_grids)}\")\n",
        "    print(f\"  Clusters數: {len(clusters)}\")\n",
        "    print(f\"  Score_i 範圍: {hh_grids['Score_i'].min():.1f} - {hh_grids['Score_i'].max():.1f}\")\n",
        "    \n",
        "    # 顯示每個cluster的統計\n",
        "    if len(clusters) > 0:\n",
        "        print(f\"\\n各 Cluster 資訊:\")\n",
        "        for idx, row in clusters.iterrows():\n",
        "            print(f\"  Cluster {row['cluster_id']}: {row['n_grids_in_cluster']}格\")\n",
        "else:\n",
        "    print(\"警告：沒有HH核心區\")\n",
        "\n",
        "# 繪製cluster邊界（不同顏色）\n",
        "if len(clusters) > 1:\n",
        "    colors = ['blue', 'cyan', 'green', 'magenta', 'orange']\n",
        "    for idx, row in clusters.iterrows():\n",
        "        color = colors[idx % len(colors)]\n",
        "        gpd.GeoSeries([row.geometry], crs=clusters.crs).boundary.plot(\n",
        "            ax=ax, color=color, linewidth=3, linestyle='--', \n",
        "            label=f'Cluster {row[\"cluster_id\"]}', zorder=5)\n",
        "\n",
        "# 繪製餐飲POI（黑色小點）\n",
        "catering_anlyzed.plot(ax=ax, color='black', markersize=2, alpha=0.4, zorder=3)\n",
        "\n",
        "# 添加街道圖底圖\n",
        "if use_basemap:\n",
        "    try:\n",
        "        cx.add_basemap(ax, crs=grid_with_kde.crs.to_string(), \n",
        "                       source=cx.providers.OpenStreetMap.Mapnik, \n",
        "                       alpha=0.6, zorder=1)\n",
        "    except Exception as e:\n",
        "        print(f\"底圖載入失敗: {e}\")\n",
        "\n",
        "# 設定標題和樣式\n",
        "ax.set_title(f'新北市三重區 - 餐飲商業區分布圖 ({len(clusters)} 個獨立商業區)\\n（深藍=最強 → 綠=中等 → 黃=較弱）', \n",
        "             fontsize=18, fontweight='bold', pad=20)\n",
        "if len(clusters) > 1:\n",
        "    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)\n",
        "ax.axis('off')\n",
        "\n",
        "# 添加比例尺\n",
        "from matplotlib.patches import Rectangle\n",
        "\n",
        "scale_length = 500  # 500公尺\n",
        "ax.add_patch(Rectangle((0.02, 0.02), 0.1, 0.01, transform=ax.transAxes, \n",
        "                       facecolor='white', edgecolor='black', linewidth=2))\n",
        "ax.text(0.07, 0.04, f'{scale_length}m', transform=ax.transAxes, \n",
        "        ha='center', va='bottom', fontsize=10, fontweight='bold',\n",
        "        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f\"\\n圖例說明:\")\n",
        "print(f\"  🔵 深藍色: 餐飲火力最強\")\n",
        "print(f\"  🟢 綠色: 餐飲火力中等\")\n",
        "print(f\"  🟡 黃色: 餐飲火力較弱\")\n",
        "print(f\"  ⬛ 黑色邊線: 100m × 100m 網格\")\n",
        "print(f\"  ⚫ 黑點: 餐飲POI位置 (共{len(catering_anlyzed)}筆)\")\n",
        "print(f\"  🔷 虛線邊界: 各獨立商業區邊界 (共{len(clusters)}個)\")\n",
        "if use_basemap:\n",
        "    print(f\"  🗺️ 底圖: OpenStreetMap 街道圖\")"
    ]
}

# 加入到 notebook 最後
nb['cells'].append(new_markdown)
nb['cells'].append(new_code)

# 寫回
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ 新增視覺化 cell 成功！")
