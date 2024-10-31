# Blake McFarlane
# Homework 3

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, 
                 num_thresholds, 
                 genuine_scores, 
                 impostor_scores, 
                 plot_title, 
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.

        Parameters:
        - num_thresholds (int): Number of thresholds to evaluate.
        - genuine_scores (array-like): Genuine scores for evaluation.
        - impostor_scores (array-like): Impostor scores for evaluation.
        - plot_title (str): Title for the evaluation plots.
        - epsilon (float): A small value to prevent division by zero.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.

        Returns:
        - float: The calculated d' value.
        """
        mu_genuine = np.mean(self.genuine_scores)
        mu_impostor = np.mean(self.impostor_scores)
        sigma_genuine = np.std(self.genuine_scores)
        sigma_impostor = np.std(self.impostor_scores)

        x = abs(mu_genuine - mu_impostor)
        y = np.sqrt(0.5 * (sigma_genuine**2 + sigma_impostor**2))
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores.
        """
        plt.figure()
        
        # Plot the histogram for genuine scores
        plt.hist(
            self.genuine_scores,      # Provide genuine scores data here
            color='green',            # color: Set the color for genuine scores
            lw=2,                     # lw: Set the line width for the histogram
            histtype='step',          # histtype: Choose 'step' for a step histogram
            hatch='/',                # hatch: Choose a pattern for filling the histogram bars
            label='Genuine Scores'    # label: Provide a label for genuine scores in the legend
        )
        
        # Plot the histogram for impostor scores
        plt.hist(
            self.impostor_scores,     # Provide impostor scores data here
            color='red',              # color: Set the color for impostor scores
            lw=2,                     # lw: Set the line width for the histogram
            histtype='step',          # histtype: Choose 'step' for a step histogram
            hatch='\\',               # hatch: Choose a pattern for filling the histogram bars
            label='Impostor Scores'   # label: Provide a label for impostor scores in the legend
        )
        
        # Set the x-axis limit to ensure the histogram fits within the correct range
        plt.xlim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Add legend to the upper left corner with a specified font size
        plt.legend(
            loc= 'upper left',
            fontsize= 12
        )
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'Matching Score',           # Provide the x-axis label
            fontsize=12,
            weight='bold'
        )
        
        plt.ylabel(
            'Score Frequency',          # Provide the y-axis label
            fontsize=12,
            weight='bold'
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set font size for x and y-axis ticks
        plt.xticks(
            fontsize=12 
        )
        
        plt.yticks(
            fontsize=12 
        )
        
        # Add a title to the plot with d-prime value and system title
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' % 
                  (self.get_dprime(), 
                   self.plot_title),
                  fontsize=15,
                  weight='bold')
        
       
        # Save the figure before displaying it
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display the plot after saving
        plt.show()
        
        # Close the figure to free up resources
        plt.close()

        return

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
    
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.
    
        Returns:
        - float: Equal Error Rate (EER).
        """
        differences = np.abs(np.array(FPR) - np.array(FNR))
        min_index = np.argmin(differences)
        EER = (FPR[min_index] + FNR[min_index]) / 2
                
        return EER

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER = self.get_EER(FPR, FNR)
        
        # Create a new figure for plotting
        plt.figure()
        
        # Plot the Detection Error Tradeoff Curve
        plt.plot(
            FPR, FNR,          # FPR values on the x-axis
            lw=2,              # lw: Set the line width for the curve
            color='blue'       # color: Set the color for the curve
        )
        
        # Add a text annotation for the EER point on the curve
        # Plot the diagonal line representing random classification
        # Scatter plot to highlight the EER point on the curve

        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100)
        
        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(
            color='gray',       # color: Set the color of grid lines
            linestyle='--',     # linestyle: Choose the line style for grid lines
            linewidth=0.5       # linewidth: Set the width of grid lines
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            'False Positive Rate',
            fontsize=12,
            weight='bold'
        )
        
        plt.ylabel(
            'False Negative Rate',
            fontsize=12,
            weight='bold'
        )
        
        # Add a title to the plot with EER value and system title
        plt.title(
            'Detection Error Tradeoff Curve \nEER = %.5f\nSystem %s' % (EER, self.plot_title),
            fontsize=15,
            weight='bold'
        )
        
        # Set font size for x and y-axis ticks
        plt.xticks(
            fontsize=12
        )
        
        plt.yticks(
            fontsize=12
        )
        
        # Save the plot as an image file
        plt.savefig(
            'DET_curve_(%s).png' % self.plot_title,
            dpi=300,
            bbox_inches='tight'
        )
        
        # Display the plot
        plt.show()        
        
        # Close the plot to free up resources
        plt.close()
    
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """
        
        # Create a new figure for the ROC curve
        plt.figure()

        # Plot the ROC curve using FPR and TPR with specified attributes
        plt.plot(
            FPR, TPR,          # FPR on x-axis, TPR on y-axis
            lw=2,              # Line width
            color='blue'       # Line color
        )

        # Set x and y axis limits, add grid, and remove top and right spines
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        plt.grid(
            color='gray',
            linestyle='--',
            linewidth=0.5
        )

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Set labels for x and y axes, and add a title
        plt.xlabel(
            'False Positive Rate',
            fontsize=12,
            weight='bold'
        )

        plt.ylabel(
            'True Positive Rate',
            fontsize=12,
            weight='bold'
        )

        plt.title(
            'Receiver Operating Characteristic Curve\nSystem %s' % self.plot_title,
            fontsize=15,
            weight='bold'
        )

        # Set font sizes for ticks, x and y labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Save the plot as a PNG file and display it
        plt.savefig(
            'ROC_curve_(%s).png' % self.plot_title,
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()
        plt.close()
 
        return

    def compute_rates(self):

        FPR, FNR, TPR = [], [], []
    
        for threshold in self.thresholds:
            FP = np.sum(self.impostor_scores >= threshold)
            TN = len(self.impostor_scores) - FP
            FN = np.sum(self.genuine_scores < threshold)
            TP = len(self.genuine_scores) - FN
            
            fpr = FP / len(self.impostor_scores)
            fnr = FN / len(self.genuine_scores)
            tpr = TP / len(self.genuine_scores)
            
            FPR.append(fpr)
            FNR.append(fnr)
            TPR.append(tpr)
    
        return FPR, FNR, TPR

def main():
    
    np.random.seed(1)

    systems = ['A', 'B', 'C']

    for system in systems:
    
        genuine_mean = np.random.uniform(0.5, 0.9)
        genuine_std = np.random.uniform(0.0, 0.2)

        genuine_scores = np.random.normal(genuine_mean, genuine_std, 400)
        
        impostor_mean = np.random.uniform(0.1, 0.5)
        impostor_std = np.random.uniform(0.0, 0.2)
        impostor_scores = np.random.normal(impostor_mean, impostor_std, 1600)

        
        # Creating an instance of the Evaluator class
        evaluator = Evaluator(
            epsilon=1e-12,
            num_thresholds=200,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            plot_title="%s" % system
        )

        FPR, FNR, TPR = evaluator.compute_rates()
    
        evaluator.plot_score_distribution()
                
        evaluator.plot_det_curve(FPR, FNR)
        
        evaluator.plot_roc_curve(FPR, TPR)
        
        
if __name__ == "__main__":
    main()

