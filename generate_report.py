from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_comparison_plot():
    # Example performance data
    metrics = {'Iteration': [1, 2, 3, 4, 5],
               'Accuracy': [0.85, 0.87, 0.89, 0.90, 0.91],
               'Confidence': [0.80, 0.82, 0.84, 0.85, 0.86]}

    df = pd.DataFrame(metrics)

    # Plot accuracy vs iteration
    plt.figure(figsize=(6, 4))
    plt.plot(df['Iteration'], df['Accuracy'], label='Accuracy')
    plt.plot(df['Iteration'], df['Confidence'], label='Confidence', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    plt.title('Performance Comparison: HITL vs Baseline')
    plt.legend()

    # Save plot as PNG
    plot_path = "performance_comparison.png"
    plt.savefig(plot_path)
    return plot_path

def create_pdf_report():
    # Create a PDF file
    pdf_filename = "HITL_Performance_Report.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(200, 750, "HITL Model Performance Report")

    # Add Introduction
    c.setFont("Helvetica", 12)
    c.drawString(72, 700, "This report compares the performance of the HITL Spam Classifier")
    c.drawString(72, 685, "against a baseline model with respect to accuracy and confidence.")

    # Generate and embed the comparison plot
    plot_path = generate_comparison_plot()
    c.drawImage(plot_path, 100, 400, width=400, height=300)

    # Performance Summary
    c.drawString(72, 370, "Summary of Model Performance: ")
    c.drawString(72, 355, "HITL model shows improved accuracy and confidence over iterations.")

    # Save PDF
    c.save()

# Generate the report
create_pdf_report()
print("PDF report generated successfully!")

