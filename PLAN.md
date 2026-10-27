# Open Insect ID v2 — Feuille de Route Techniques & Spécifications Architecture

 Ce document constitue la spécification technique et la feuille de route opérationnelle pour la création de la version 2 du projet **Open Insect ID**. La v2 repose sur une architecture faite maison de type **Vision Mixture of Experts (MoE)** organisée en cascade taxonomique à 4 niveaux, entraînée sur les données globales du GBIF et optimisée pour l'inférence *on-device* (Android/Kotlin).

---

## 1. Sourcing et Pipeline de Données (GBIF & Metadata Engine)

### 1.1 Extraction et Sourcing via l'API GBIF
* **Périmètre ciblée :** Extraction de la faune entomologique européenne / continentale pour concilier représentativité et volumétrie maîtrisée.
* **Filtres de requêtes :**
  * `taxonKey` : Hexapoda / Insecta.
  * `mediaType` : StillImage.
  * `occurrenceStatus` : PRESENT.
  * `basisOfRecord` : HUMAN_OBSERVATION, PRESERVED_SPECIMEN (privilégier les validations iNaturalist / muséums).
* **Extraction des métadonnées taxonomiques :** Pour chaque enregistrement, extraction explicite de la lignée stricte :
  $$\text{Ordre} \longrightarrow \text{Famille} \longrightarrow \text{Genre} \longrightarrow \text{Espèce}$$

### 1.2 Ingestion & Téléchargement Asynchrone
* Développement d'un pipeline Python d'ingestion (`asyncio` + `aiohttp`) gérant la tolérance aux pannes, les timeouts, et la dédoublonnisation d'URLs.
* Stockage structuré sur disque : `data/raw/{ordre}/{famille}/{genre}/{espece}/{image_hash}.jpg`.

### 1.3 Data Curation et Nettoyage Automatisé
1. **Validation d'intégrité visuelle :** Suppression des images corrompues, résolutions inférieures à $224 \times 224$, ou ratios d'aspect extrêmes.
2. **Filtrage sémantique par Embeddings :** Utilisation de **BioCLIP** (ou CLIP de base) pour calculer la cohérence vectorielle des images avec la classe d'insectes attendue et éliminer les faux positifs (photos de paysages, flous de mouvement sans insecte, artefacts).
3. **Stratégie pour la Longue Traîne (*Long-Tail Distribution*) :**
   * *Sous-échantillonnage intelligent* des espèces surreprésentées (plafonnement du nombre d'images par espèce).
   * *Data Augmentation conditionnelle* pour les classes rares : Mixup, CutMix, variations chromatiques (simulant la lumière naturelle en sous-bois) et déformations géométriques.

---

## 2. Architecture du Modèle : Vision MoE Hiérarchique à 4 Niveaux

Le système abandonne la classification globale standard au profit d'un arbre d'experts spécialisés, réduisant la complexité de chaque sous-réseau.

```
                    [Image d'entrée]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [Routeur Niveau 1 : Ordre] ──► Confiance < Seuil ──► [Incertitude Niveau Ordre]
             │ (Top-1 / Top-2)
             ├───────────────────────────┐
             ▼                           ▼
   [Expert Niveau 2 : Famille A]   [Expert Niveau 2 : Famille B]
             │                           │
             └─────────────┬─────────────┘
                           ▼
              [Expert Niveau 3 : Genre]
                           │
                           ▼
             [Expert Niveau 4 : Espèce]
```

### 2.1 Les 4 Couches Taxonomiques
1. **Niveau 1 — Routeur Macro (Ordre) :** Classifier global léger entraîné pour séparer les grands ordres (Coléoptères, Lépidoptères, Hyménoptères, Diptères, Hétéroptères...).
2. **Niveau 2 — Experts Ordre (Famille) :** Réseaux spécialisés, un par grand ordre. Chaque expert n'apprend que la discrimination fine entre les familles appartenant à cet ordre.
3. **Niveau 3 — Experts Famille (Genre) :** Réseaux spécialisés au niveau de la famille pour distinguer les genres.
4. **Niveau 4 — Micro-Experts Genre (Espèce) :** La couche finale discriminant les espèces au sein d'un même genre.

### 2.2 Sélection des Backbones
* **Squelette principal (Edge ML) :** **MobileNetV4 (Small / Medium)** avec blocs UIB (*Universal Inverted Bottleneck*). Choisi pour son ratio précision/latence supérieur sur CPU/NPU mobile par rapport à MobileNetV3.
* **Transfer Learning & Transfert de Poids (Feature Extractor) :**
  * Initialisation des poids à partir d'un backbone **BioCLIP** (pré-entraîné sur l'arbre du vivant iNaturalist) pour transférer la sensibilité aux caractéristiques visuelles biologiques (stries élytrales, nervures d'ailes, pièces buccales).
  * Distillation de connaissances (*Knowledge Distillation*) du modèle BioCLIP grand format vers les modules MobileNetV4 compacts.

---

## 3. Stratégie d'Entraînement et Fonctions de Perte

### 3.1 Hierarchical Cross-Entropy Loss (Pénalité Taxonomique)
Pour éviter de traiter une erreur sur l'ordre de la même manière qu'une erreur sur l'espèce, la fonction de perte intègre la distance $d$ dans l'arbre taxonomique :

$$\mathcal{L}_{\text{hierarchical}} = \sum_{k=1}^{4} w_k \cdot \mathcal{L}_{\text{CE}}(y_k, \hat{y}_k) + \lambda \cdot \text{Cost Matrix}(y_{\text{pred}}, y_{\text{true}})$$

* Se tromper d'Ordre inflige une pénalité maximale ($\lambda$ élevé).
* Se tromper d'Espèce au sein du bon Genre est pénalisé de façon minimale.

### 3.2 Loss Adaptée au Déséquilibre : Focal Loss
Pour les derniers niveaux d'experts (Genres et Espèces), remplacement de la Cross-Entropy standard par la **Focal Loss** afin de concentrer les gradients sur les exemples difficiles et éviter la saturation due aux classes fréquentes :

$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

---

## 4. Stratégie de Routage, Incertitude et Repli

### 4.1 Top-K Routing et Gestion de Cascade
Pour éviter l'erreur de cascade (où un mauvais choix au Niveau 1 condamne toute la prédiction) :
* Si la probabilité du Top-1 du Routeur est sous un seuil $\tau_1$ (ex: $< 0.85$), le pipeline conserve le **Top-2** des ordres.
* L'image est transmise aux deux experts Famille correspondants, et leurs scores de confiance sont combinés de manière pondérée.

### 4.2 Rejet et Détection d'Incertitude
* À chaque niveau d'expert, la distribution de probabilité softmax est évaluée via l'entropie de prédiction.
* Si le niveau de confiance tombe sous le seuil d'acceptation $\tau_{\text{level}}$, le pipeline s'arrête prématurément et la réponse remonte au niveau taxonomique le plus bas validé avec certitude :
  * *Exemple :* *"Espèce incertaine. Identifié avec certitude comme appartenant au Genre : Carabus (Famille : Carabidae)"*.

---

## 5. Exportation et Optimisation Edge ML (TFLite)

1. **Quantification Post-Entraînement (PTQ) & Quantization-Aware Training (QAT) :**
   * Conversion des poids du modèle PyTorch de `FP32` vers `INT8` avec un jeu de données de calibration.
   * Gain : Réduction par 4 de l'empreinte mémoire et accélération d'inférence via NNAPI / Hexagon DSP.
2. **Exportation multi-graphes ou sous-graphes :**
   * Export au format **ONNX** puis compilation en fichiers `.tflite`.
   * Deux stratégies de packaging :
     * *Option A :* Modèles `.tflite` distincts stockés dynamiquement ou chargés sur demande dans l'application.
     * *Option B :* Graphe unifié TFLite avec opérateurs de contrôle de flux.

---

## 6. Déploiement Mobile Native (Android / Kotlin)

* **Architecture Android :** Architecture MVVM / Clean Architecture avec Coroutines et Kotlin Flow.
* **Pipeline d'Inference Local :**
  * Utilisation du SDK `org.tensorflow:tensorflow-lite`.
  * Exécution asynchrone hors du Thread UI pour maintenir le framerate de la caméra.
  * Gestionnaire de mémoire (Buffer pools) pour éviter les fuites de RAM lors de l'exécution successive de plusieurs experts MoE.
* **Interface Utilisateur (UI) :**
  * Visualisation dynamique de l'arbre taxonomique lors du scan (progression du routage en temps réel : Ordre $\rightarrow$ Famille $\rightarrow$ Genre $\rightarrow$ Espèce).
  * Affichage explicite de l'indice de certitude avec mise en valeur du dernier niveau validé.

---

## 7. Structuration du Projet Open Source (Kaggle & GitHub)

1. **Dataset Kaggle v2 :**
   * Publication du dataset filtré et nettoyé, structuré avec métadonnées JSON complètes.
   * Score d'usabilité visé : `10/10` (Fiche descriptive complète, schémas d'arborescence, notebooks d'exemples).
2. **Dépôt GitHub :**
   * Module `src/data_pipeline` : Scripts asynchrones d'ingestion GBIF et de curation.
   * Module `src/models` : Implémentation PyTorch du Vision MoE et des Custom Losses.
   * Module `src/export` : Scripts de quantification et de conversion TFLite.
   * Application `android/` : Code source Kotlin de l'application mobile.
   * Documentation : README détaillé avec schémas d'architecture et benchmarks de latence.
   
   
   
   
   
   # Open Insect ID v2 — Spécifications Techniques & Plan d'Ingénierie

## 1. Vue d'Ensemble & Objectifs
* **Périmètre :** Monde entier (World-scale taxonomy).
* **Objectif :** Concevoir une application Android native (Kotlin) d'identification d'insectes reposant sur une architecture de vision par ordinateur en cascade (MoE hiérarchique PyTorch à 4 niveaux) tournant en inférence locale (TFLite/ONNX).
* **Innovation Clé :** Fusion multimodal de la vision (*Edge MoE*) et de la biogéographie (*Geofence Prior*) pour éliminer les ambiguïtés morphologiques et nettoyer le dataset.

---

## 2. Pipeline de Données : Acquisition & Nettoyage Mondial (GBIF & DuckDB)

### A. Requête d'Acquisition Asynchrone (GBIF Occurrence API)
* **Filtres de téléchargement mondial :**
  - `classKey`: 216 (Insecta)
  - `hasMultimedia`: `true`
  - `hasCoordinate`: `true`
  - `issue`: Exclure `COORDINATE_INVALID`, `ZERO_COORDINATE`, `COUNTRY_COORDINATE_MISMATCH`.
* **Export :** Archive unique compressed Parquet/TSV (plusieurs dizaines de millions de lignes).

### B. Traitement Haute Performance (DuckDB / Polars)
* **Streaming sans saturation RAM :** Utilisation de DuckDB en Python pour requêter le fichier TSV géant directement sur disque.
* **Indexation Spatiale (Uber H3 Indexing) :**
  - Découpage de la surface terrestre en cellules hexagonales H3 (résolution 4 ou 5).
  - Construction d'une matrice de présence/absence par espèce basée sur la densité d'occurrences.
* **Filtrage des Outliers Spatiaux (Data Curation) :**
  - Élimination automatique des images dont les coordonnées GPS tombent hors du masque de distribution H3 connu de l'espèce (erreurs de saisie, spécimens de laboratoire).
* **Ingestion Asynchrone des Images :**
  - Script Python (`asyncio` + `aiohttp`) pour télécharger le pool d'images nettoyé avec rate-limiting par domaine CDN (iNaturalist, Muséums).

---

## 3. Architecture du Modèle : Vision MoE Hiérarchique à 4 Niveaux

### A. Structure en Cascade (Routage Taxonomique)
1. **Niveau 1 — Routeur Macro (Ordre) :** Backbone ultra-léger classant l'image parmi les grands ordres d'insectes.
2. **Niveau 2 — Experts Famille :** Modèles spécialisés dédiés à un ordre spécifique.
3. **Niveau 3 — Experts Genre :** Affinement au sein de la famille validée.
4. **Niveau 4 — Micro-Experts Espèce :** Classification fine des espèces morphologiquement proches.

### B. Choix des Backbones & Entraînement
* **Backbones Cibles :** **MobileNetV4** (Conv-Small / Medium) pour l'efficacité NPU/CPU mobile, ou distillation depuis **BioCLIP** (pré-entraîné sur iNaturalist).
* **Fonction de Perte (Hierarchical Tree Loss & Focal Loss) :**
  - Pénalité maximale appliquée si l'erreur concerne un niveau supérieur (ex: mauvais Ordre).
  - *Focal Loss* pour équilibrer la longue traîne (*long-tail distribution*) des espèces rares.

---

## 4. Inférence Hybride : Vision + Geofence Prior (Sur Mobile)

### A. Modélisation Biogéographique en Local
* **Table Spatial Hash Embarquée :** Une base SQLite/KVP compacte associée à l'application Kotlin stockant le masque H3 des espèces par région du monde.

### B. Fusion Probabiliste Bayésienne
Lors de la prise de vue par l'utilisateur :
$$\text{Score Final} = P(\text{Espèce} \mid \text{Image}) \times P(\text{Espèce} \mid \text{GPS})$$
* **Résolution des ambiguïtés :** Si deux espèces sont visuellement identiques, les coordonnées GPS tranches immédiatement en fonction de la région écologique locale.
* **Gestion d'Incertitude (Fallback) :** Arrêt du pipeline au niveau taxonomique le plus fiable (ex: "Genre certain, Espèce incertaine") si la confiance combinée passe sous un seuil paramétré.

---

## 5. Deployment & Open Source
* **Quantification & Export :** Quantification `INT8` Post-Training et conversion TFLite / ONNX.
* **Integration Kotlin :** Inférence multithreadée avec la caméra Android.
* **Publication :** Dataset propre mondial sur Kaggle (Target Usability 10/10) & Repository GitHub structuré.
