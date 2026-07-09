args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1] 
output_dir <- args[2]

library(dplyr)
library(ggplot2)

# ============Step1:Load abundance and cauchy data=====================================================
df <- read.csv(input_file)


# ============Step2:Vis bubble=====================================================
df$log10P_category <- cut(
  df$neg_log10_p,
  breaks = c(-Inf, 1, 5, 10, 11, Inf),   
  labels = c("log10P <= 1", "1 < log10P <= 5", "5 < log10P <= 10","10 < log10P <= 11", "log10P > 11")
)


color_map = c("#F0F0F0", "#C6E6E8", "#93D5DC","#51C4D3", "#10AEC2", "#0F95B0")

p <- ggplot(df, aes(x = type, y = pathway, color = comm_count, size = log10P_category)) +
  geom_point(data = subset(df, comm_count > 0),alpha = 1) +
  scale_color_gradientn(
    colors = color_map, 
    name = "Abundance"
  ) +
  scale_size_manual(
    values = c(
      "log10P <= 1" = 3, 
      "1 < log10P <= 5" = 5, 
      "5 < log10P <= 10" = 7, 
      "10 < log10P <= 11" = 10, 
      "log10P > 11" = 15
    ),
    name = "-log10(P-value)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    axis.title = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    legend.position = "right"
  ) +
  labs(
    title = "Signaling Pathway Abundance by Cell Type",
    caption = "Point color: Pathway abundance\nPoint size: Pathway abundance"
  )

print(p)


output_png <- file.path(output_dir, "CauchyP_Dot_plot.png")  
ggsave(output_png, plot = p, width = 8, height = 12, dpi = 300)  
cat("Plot saved to:",  output_png, "\n")





