spec_version: "0.1"
spec_name: "商業分析流程-評估流程（可程式化SOP規格）"
language: "zh-Hant"
vault_root: "c:/obsidian-repo/knowledge-base"
generated_from:
  entry_card:
    title: "LIT-25A1D1-商業分析流程-評估流程"
    path: "02_Literature/LIT-25A1D1-商業分析流程-評估流程.md"
  linked_cards:
    - title: "LIT-25A1B1-商業分析流程-餐飲POI"
      path: "02_Literature/LIT-25A1B1-商業分析流程-餐飲POI.md"
    - title: "LIT-25A1-商圈分析流程"
      path: "02_Literature/LIT-25A1-商圈分析流程.md"
    - title: "LIT-25A1C1-商業分析流程-百米方格"
      path: "02_Literature/LIT-25A1C1-商業分析流程-百米方格.md"
    - title: "AI-CODEX-FC-0-LIT-25A1B1"
      path: "03_AI_Response/AI-CODEX-FC-0-LIT-25A1B1.md"
    - title: "AI-CODEX-FC-00-LIT-25A1D1"
      path: "03_AI_Response/AI-CODEX-FC-00-LIT-25A1D1.md"
    - title: "?有怎?的商圈，城市才有更好的?展机?"
      path: "99_Raw/?有怎?的商圈，城市才有更好的?展机?.md"
  sources:
    - type: "note_or_asset_link"
      title: "擁有怎麼樣的商圈_城市才有更好的發展.pdf"
    - type: "note_or_asset_link"
      title: "基於poi數據分析發現商業中心方法.pdf"
    - type: "note_or_asset_link"
      title: "基於poi數據分析發現商業中心方法_2.pdf"
    - type: "note_or_asset_link"
      title: "基於poi數據分析發現商業中心方法_3.pdf"

obsidian_linking:
  wiki_link_pattern: "[[...]]"
  resolution_rules:
    - "以 link_text 對應同名 md 檔（可跨資料夾）"
    - "若同名多檔，需額外提供 folder_hint 或唯一ID規則"
  this_spec_scope: "僅涵蓋 entry_card 直接連結之卡片（不遞迴展開）"

domain:
  problem_statement: "以餐飲POI與網格為基礎，透過 KDE 平滑 + LISA（Local Moran’s I，含距離衰減）識別商業區，並將商業區分級成商圈（circles）。"
  target_hierarchy: ["POI", "Grid", "Cluster", "Circle"]
  intended_use_cases:
    - "產出可重現的商圈（圈）polygon與等級"
    - "對流程參數做敏感度/穩健性報表，用於方法辯護"

data_contracts:
  crs:
    required: true
    default: "EPSG:3826"
    distance_unit: "meter"
  inputs:
    poi:
      geometry_type: "Point"
      required_fields:
        poi_id: {type: "string|int"}
        name: {type: "string"}
        category: {type: "string"}
        geometry: {type: "geometry(Point)"}
      optional_fields:
        name_normalized: {type: "string"}
        purpose_type: {type: "enum", values: ["目的型", "補給型"]}
        chain_flag: {type: "enum", values: ["chain", "non_chain", "unknown"]}
        chain_store_count: {type: "int"}
    grid:
      geometry_type: "Polygon"
      required_fields:
        grid_id: {type: "string|int"}
        geometry: {type: "geometry(Polygon)"}
      derived_fields:
        centroid: {type: "geometry(Point)"}
  outputs:
    grid_raw:
      key: ["grid_id"]
      fields:
        grid_id: {type: "string|int"}
        raw_i: {type: "float"}
    grid_kde:
      key: ["grid_id"]
      fields:
        grid_id: {type: "string|int"}
        D_i: {type: "float"}
        Score_i: {type: "float", range: [0, 100]}
    grid_lisa:
      key: ["grid_id"]
      fields:
        grid_id: {type: "string|int"}
        z_i: {type: "float"}
        lag_z_i: {type: "float"}
        I_i: {type: "float"}
        p_value: {type: "float"}
        q_value: {type: "float", nullable: true}
        quadrant: {type: "enum", values: ["HH", "LL", "HL", "LH"]}
        HH_flag: {type: "bool"}
    clusters:
      geometry_type: "Polygon"
      key: ["cluster_id"]
      fields:
        cluster_id: {type: "string|int"}
        N_grids: {type: "int"}
        TotalPower: {type: "float"}
        Peak: {type: "float"}
        Area: {type: "float"}
        geometry: {type: "geometry(Polygon)"}
    circles:
      geometry_type: "Polygon"
      key: ["cluster_id"]
      fields:
        cluster_id: {type: "string|int"}
        tier: {type: "enum", values: ["Tier1", "Tier2", "Non-Circle"]}
        N_grids: {type: "int"}
        TotalPower: {type: "float"}
        Peak: {type: "float"}
        Area: {type: "float"}
        geometry: {type: "geometry(Polygon)"}
    sensitivity_report:
      key: ["run_id"]
      fields:
        run_id: {type: "string"}
        params: {type: "object"}
        circle_count: {type: "int"}
        topN_jaccard: {type: "float"}
        rank_spearman: {type: "float"}
        hh_grid_count: {type: "int"}
        cluster_count: {type: "int"}
        primary_variation_driver: {type: "string"}

config:
  grid:
    grid_size_m:
      default: 300
      notes: "實際以既有網格為準；本流程只假設是完整等面積網格"
  poi_modeling:
    dedupe_key: ["name_normalized", "geometry"]
    scoring:
      formulae:
        s_k_i: "w_k * log(1 + min(count_{k,i}, cap_k))"
        raw_i: "sum_k s_{k,i}"
      weight_principles: ["目的型 > 補給型 > 易灌水類"]
      cap_policy:
        apply_to_categories: ["手搖飲店", "其他飲料店", "調理飲料攤販", "餐食攤販"]
        default_cap: 10
  poi_categories:
    source: "02_Literature/LIT-25A1B1-商業分析流程-餐飲POI.md"
    categories:
      - {name: "便當、自助餐店", purpose_type: "補給型"}
      - {name: "其他飲料店", purpose_type: "補給型", high_density_cap: true}
      - {name: "吃到飽餐廳", purpose_type: "目的型"}
      - {name: "咖啡館", purpose_type: "目的型"}
      - {name: "手搖飲店", purpose_type: "補給型", high_density_cap: true}
      - {name: "早餐店", purpose_type: "補給型"}
      - {name: "有娛樂節目餐廳", purpose_type: "目的型"}
      - {name: "茶館", purpose_type: "目的型"}
      - {name: "調理飲料攤販", purpose_type: "補給型", high_density_cap: true}
      - {name: "連鎖速食店", purpose_type: "補給型"}
      - {name: "飲酒店", purpose_type: "目的型"}
      - {name: "餐廳", purpose_type: "目的型"}
      - {name: "餐食攤販", purpose_type: "補給型", high_density_cap: true}
      - {name: "麵店、小吃店", purpose_type: "補給型"}
    chain_flag:
      method_hint: "店名標準化 + 全域出現頻次門檻 → chain/non_chain/unknown"
  kde:
    unit_of_analysis: "grid centroid"
    input_mode:
      default: "A"
      options:
        A: {source_points: "grid centroids", weights: "raw_i"}
        B: {source_points: "poi points", weights: null}
    bandwidth_h_m:
      default: 600
      sensitivity_values: [450, 600]
    output_standardization:
      score_method: "rank_pct"
      score_range: [0, 100]
  lisa:
    variable: "Score_i"
    zscore: true
    weights_matrix:
      distance_threshold_d0_m:
        default: 900
        sensitivity_values: [600, 900, 1200]
      decay:
        formula: "w_ij = 1 / d_ij^alpha"
        alpha:
          default: 1
          sensitivity_values: [1, 2]
      standardization: "row-standardize"
    inference:
      permutations: 999
      multiple_testing: {method: "BH-FDR", enabled: true}
    hh_filter:
      rules:
        - 'quadrant == "HH"'
        - "z_i > 0"
        - "lag_z_i > 0"
        - "q_value < 0.05"
      mvp_fallback:
        use_p_value: true
        p_value_threshold: 0.05
        note_required: "需標註未做BH-FDR校正"
  clustering:
    method: "dissolve contiguous HH grids"
    adjacency: "spatial contiguity"
    cluster_attributes:
      N_grids: "count(grids)"
      TotalPower: "sum(Score_i)"
      Peak: "max(Score_i)"
      Area: "area(cluster geometry)"
  circle_tiering:
    tier1:
      rules:
        - "N_grids >= 10"
        - "TotalPower >= P90(TotalPower)"
    tier2:
      rules:
        - "N_grids >= 4"
        - "TotalPower >= mean(TotalPower)"
    else: "Non-Circle"
  sensitivity:
    required: true
    parameter_matrix_minimum:
      kde_bandwidth_h_m: [450, 600]
      lisa_d0_m: [600, 900, 1200]
      lisa_alpha: [1, 2]
      weights_and_caps_perturbation: {enabled: true, delta_pct: 20}
    stability_metrics:
      - {name: "topN_jaccard", description: "Top-N circles 重疊率"}
      - {name: "rank_spearman", description: "circles 排名相關"}
      - {name: "counts_delta", description: "HH格子數與cluster數量變化"}

pipeline:
  steps:
    - id: "S1_preprocess_poi"
      title: "資料前處理（清洗/去重/分層）"
      inputs: ["poi"]
      outputs: ["poi_clean"]
      actions:
        - "filter categories to poi_categories"
        - "deduplicate by dedupe_key"
        - "derive purpose_type (optional but recommended)"
        - "derive chain_flag (optional)"
    - id: "S2_grid_join_and_raw"
      title: "POI → Grid 空間對位與 raw_i 計分"
      inputs: ["poi_clean", "grid"]
      outputs: ["grid_raw"]
      actions:
        - "spatial_join predicate=within"
        - "aggregate count_{k,i} by grid_id and category k"
        - "compute raw_i using scoring formulae"
    - id: "S3_kde"
      title: "KDE 平滑（Grid Density）"
      inputs: ["grid_raw", "grid", "poi_clean"]
      outputs: ["grid_kde"]
      actions:
        - "derive centroid_i from grid"
        - "run KDE using input_mode (A or B)"
        - "output D_i and Score_i (rank_pct 0-100)"
    - id: "S4_lisa"
      title: "LISA（Local Moran’s I，含距離衰減）"
      inputs: ["grid_kde", "grid"]
      outputs: ["grid_lisa"]
      actions:
        - "compute z_i = zscore(Score_i)"
        - "build W with d0 and distance decay; row-standardize"
        - "compute Local Moran’s I and quadrant"
        - "permutation test; BH-FDR to q_value"
        - "apply HH filter → HH_flag"
    - id: "S5_clusters"
      title: "HH grids → clusters"
      inputs: ["grid_lisa", "grid_kde", "grid"]
      outputs: ["clusters"]
      actions:
        - "select grids where HH_flag == true"
        - "dissolve contiguous polygons"
        - "compute cluster attributes"
    - id: "S6_circles"
      title: "clusters → circles 分級"
      inputs: ["clusters"]
      outputs: ["circles"]
      actions:
        - "apply tier rules (Tier1/Tier2/Non-Circle)"
    - id: "S7_sensitivity"
      title: "敏感度/穩健性報表"
      inputs: ["pipeline", "parameter_matrix_minimum"]
      outputs: ["sensitivity_report"]
      actions:
        - "rerun S3-S6 over parameter grid"
        - "compute stability_metrics and summarize primary driver"

card_index:
  - path: "02_Literature/LIT-25A1D1-商業分析流程-評估流程.md"
    role: ["canonical_sop"]
    key_links:
      - "LIT-25A1B1-商業分析流程-餐飲POI"
      - "LIT-25A1-商圈分析流程"
      - "LIT-25A1C1-商業分析流程-百米方格"
      - "AI-CODEX-FC-0-LIT-25A1B1"
      - "?有怎?的商圈，城市才有更好的?展机?"
  - path: "02_Literature/LIT-25A1B1-商業分析流程-餐飲POI.md"
    role: ["poi_taxonomy_and_weights"]
  - path: "02_Literature/LIT-25A1C1-商業分析流程-百米方格.md"
    role: ["grid_build_guidance"]
  - path: "02_Literature/LIT-25A1-商圈分析流程.md"
    role: ["project_goal_and_decomposition"]
  - path: "03_AI_Response/AI-CODEX-FC-00-LIT-25A1D1.md"
    role: ["ai_sop_copy_for_traceability"]
  - path: "03_AI_Response/AI-CODEX-FC-0-LIT-25A1B1.md"
    role: ["ai_weighting_rationale_and_sensitivity_emphasis"]
  - path: "99_Raw/?有怎?的商圈，城市才有更好的?展机?.md"
    role: ["method_origin_reference"]
    note: "文字編碼可能非UTF-8；本規格僅引用其方法要點（KDE + LISA + 商圈定義）。"
