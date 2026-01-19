import networkx as nx
import matplotlib.pyplot as plt

#--------------------------------------------
# Explanation of Steps and Integration of Numerical Data
#--------------------------------------------
# This code creates two separate directed graphs:
# 1) Research graph: Each author node from the "Research" Papers connects to method nodes.
#    Methods represent techniques or tools the authors used or developed.
#    For example, "Kim_R" performed "High_throughput_mIHC" on "513 patient samples".
#    We include this information in the node labels or in code comments.
#
# 2) Review graph: Each author node from the "Review" Papers connects to conceptual or methodological frameworks they discussed.
#    For instance, "ZhangZ_V" covers "3D_bioprinted_tumor_models" for immunotherapy resistance.
#
# The "_R" suffix is used for Research authors and "_V" for Review authors.
#
# Numerical data citations (e.g., "513 patient samples" for Kim_R, "70-80% LAG3+ T cells" for Zahraeifard_R,
# "966 samples" and "7 glycolysis genes" for Dai_R, "r=0.612" correlation for Sun_R, and "69 TIDE marker genes" for Zheng_R)
# are mentioned in comments next to their edges or nodes. This helps trace back to their significance.
#
#--------------------------------------------

#--------------------------------------------
# Research Graph
#--------------------------------------------
research_authors = [
    "Lahusen_R", "Kim_R", "Hu_R", "Imran_R", "Zahraeifard_R",
    "FerriB_R", "Tang_R", "Dai_R", "Sun_R", "Zheng_R"
]

research_methods = {
    "Lahusen_R": ["InterOMaX_3D_Platform", "T_cell_infiltration_analysis", "CXCL17_gene_validation"],
    # Lahusen et al. introduced InterOMaX for PDAC and validated CXCL17 mediating T cell resistance
    "Kim_R": ["High_throughput_mIHC", "AhR_expression_clustering"],
    # Kim et al.: High-throughput mIHC on "513 patient samples"
    "Hu_R": ["INHBA_CAF_identification", "Single_cell_scRNAseq_TME"],
    # Hu et al.: single-cell analysis to identify INHBA+ CAFs enriching Treg infiltration
    "Imran_R": ["Irreversible_electroporation_IRE", "Flow_cytometry_immune_profiling"],
    # Imran et al.: IRE in pancreatic cancer and flow cytometry over time
    "Zahraeifard_R": ["CRISPR_in_vivo_screen_TSGs", "LAG3_T_cell_mediated_suppression"],
    # Zahraeifard et al.: Identified NF1, TSC1, TGFβRII and found "70-80% LAG3+ T cells"
    "FerriB_R": ["3D_MSI_Stereo_seq_integration", "Metabolite_glycan_peptide_mapping"],
    # Ferri-B et al.: Integration of MSI, Stereo-seq and seqIF in 3D
    "Tang_R": ["SPP1_TAM_subpopulations", "Glioma_TAM_landscape"],
    # Tang et al.: Identified SPP1+ TAMs in glioma
    "Dai_R": ["Glycolysis_subtype_prognosis", "Spatial_transcriptomics_DLBCL"],
    # Dai et al.: "966 samples", "7 glycolysis genes" prognostic model in DLBCL
    "Sun_R": ["HIF_1alpha_PD_L1_axis", "CoCl2_DFO_hypoxia_model"],
    # Sun et al.: "r=0.612" correlation between HIF-1α and PD-L1 in CRC
    "Zheng_R": ["TIDE_based_subtyping", "69_TIDE_marker_genes"]
    # Zheng et al.: "69 TIDE marker genes" to classify bladder cancer immunotherapy response
}

G_research = nx.DiGraph()

# Add author nodes and method nodes
for author, methods in research_methods.items():
    G_research.add_node(author, color='red', type='author')
    for m in methods:
        G_research.add_node(m, color='blue', type='method')
        G_research.add_edge(author, m)

#--------------------------------------------
# Review Graph
#--------------------------------------------
review_authors = [
    "ZhangZ_V", "ZhouZ_V", "Safaei_V", "ZhangH_V", "Kundu_V",
    "Lu_V", "Li_V", "Han_V", "Du_V", "Tian_V"
]

# Concepts / frameworks discussed in reviews
review_concepts = {
    "ZhangZ_V": ["3D_bioprinted_tumor_models", "Overcoming_immunotherapy_resistance"],
    "ZhouZ_V": ["Multi_omics_integration", "CAR_T_cell_optimization"],
    "Safaei_V": ["Exosome_immune_interplay"],
    "ZhangH_V": ["Metabolic_reprogramming_in_TME"],
    "Kundu_V": ["CAF_and_TAM_modulation"],
    "Lu_V": ["Immunosuppressive_cell_subsets"],
    "Li_V": ["Treg_heterogeneity_and_anti_tumor"],
    "Han_V": ["m6A_epitranscriptomic_regulation"],
    "Du_V": ["Overcoming_immunotherapy_resistance"],  # overlaps with ZhangZ_V concept
    "Tian_V": ["Single_cell_informatics_TME", "Spatial_transcriptomics_analysis"]
}

G_review = nx.DiGraph()

for author, concepts in review_concepts.items():
    G_review.add_node(author, color='red', type='author')
    for c in concepts:
        G_review.add_node(c, color='green', type='concept')
        G_review.add_edge(author, c)

#--------------------------------------------
# Visualization
#--------------------------------------------

def draw_graph(G, title):
    pos = nx.spring_layout(G, seed=42)  # fixed seed for reproducibility
    # Extracting node colors
    node_colors = [G.nodes[n].get('color', 'gray') for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, font_size=8, node_size=1000, arrows=True)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()

# Draw the research graph
draw_graph(G_research, "Research Papers Ontological Knowledge Graph")

# Draw the review graph
draw_graph(G_review, "Review Papers Ontological Knowledge Graph")
