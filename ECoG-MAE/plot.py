import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


def plot_signal(args, signal):

    dir = os.getcwd() + f"/results/signal"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # plotting
    pdf_pages = PdfPages(dir + f"{args.job_name}_signal.pdf")
    colors = ["g", "r", "b", "c", "m"]

    for h in range(1, 8):
        for w in range(1, 8):

            fig, axs = plt.subplots(len(args.bands), 1, figsize=(8, 10))

            for i, ax in enumerate(axs):

                ax.plot(
                    np.squeeze(
                        np.reshape(
                            signal,
                            (args.batch_size * args.new_fs * args.sample_length, -1),
                        )
                    ),
                    color="b",
                    lw=0.25,
                )

            # Set title for the entire PDF page
            fig.suptitle(f"G" + str(h * h + w), fontsize=16)

            # Adjust layout and save the subplot to the PDF file
            fig.tight_layout(
                rect=[0, 0.03, 1, 0.95]
            )  # Add a little space at the top for the title
            pdf_pages.savefig(fig)
            plt.close(fig)

    # Close the PDF file
    pdf_pages.close()


def plot_losses(args, recon_losses, test_losses):

    plt.figure(figsize=(8, 3))
    plt.plot(recon_losses)
    plt.title("Training re-construction losses")
    dir = os.getcwd() + f"/results/loss/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + f"{args.job_name}_training_loss.png")

    plt.figure(figsize=(8, 3))
    plt.plot(test_losses)
    plt.title("Test losses")
    dir = os.getcwd() + f"/results/loss/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + f"{args.job_name}_test_loss.png")


def plot_contrastive_loss(args, contrastive_losses):
    plt.figure(figsize=(8, 3))
    plt.plot(contrastive_losses)
    plt.title("Training contrastive losses")
    dir = os.getcwd() + f"/results/loss/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + f"{args.job_name}_contrastive_loss.png")


def plot_correlation(args, df, fn):

    dir = os.getcwd() + f"/results/correlation/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    groups = df.groupby(["elec", "band"])
    df["x"] = groups.cumcount()

    # plotting
    pdf_pages = PdfPages(dir + f"{args.job_name}_{fn}.pdf")

    colors = {"theta": "g", "alpha": "r", "beta": "b", "gamma": "c", "highgamma": "m"}

    elecs = df.groupby("elec")

    subplot_height_ratios = [1, 1, 1, 1, 1]

    # Iterate over each elec group
    for key, elec in elecs:
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=subplot_height_ratios)

        # Iterate over each band within the elec group
        for i, (_, band_group) in enumerate(elec.groupby("band")):
            ax = fig.add_subplot(gs[i, 0])

            # Plot 'corr' against 'x' for the band
            ax.plot(
                band_group["x"],
                band_group["corr"],
                label=f'{band_group["band"].iloc[0]}',
                color=colors[band_group["band"].iloc[0]],
                lw=0.25,
            )

            # Set title on the right side
            ax.set_title(f'{band_group["band"].iloc[0]}', loc="right")

            # Hide x ticks and labels on all but the bottom subplot
            if i < len(colors) - 1:
                ax.set_xticks([])
                ax.set_xticklabels([])

            # Set x axis label only on the bottom subplot
            if i == len(colors) - 1:
                ax.set_xlabel("Iteration")

            ax.set_ylabel("Corr (r)")

            # ax.legend()

        # Set title for the entire PDF page
        fig.suptitle(f"G{key}", fontsize=16)

        # Adjust layout and save the subplot to the PDF file
        fig.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Add a little space at the top for the title
        pdf_pages.savefig(fig)
        plt.close(fig)

    # Close the PDF file
    pdf_pages.close()


def plot_recon_signals(args, df):

    dir = os.getcwd() + f"/results/recon_signals/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # plotting
    pdf_pages = PdfPages(dir + f"{args.job_name}_recon_signals.pdf")
    elecs = df.groupby("elec")

    for key, elec in elecs:

        fig, axs = plt.subplots(int(np.ceil(args.num_epochs / 2)), 2, figsize=(10, 15))

        axs = axs.flatten()

        for e, ax in zip(elec.epoch, axs):

            ax.plot(
                np.linspace(0, len(elec.iloc[e, :].x[0]), len(elec.iloc[e, :].x[0])),
                elec.iloc[e, :].x[0],
                label="original",
                color="b",
                lw=0.25,
            )
            ax.plot(
                np.linspace(0, len(elec.iloc[e, :].y[0]), len(elec.iloc[e, :].y[0])),
                elec.iloc[e, :].y[0],
                label="reconstructed",
                color="r",
                lw=0.25,
            )

            ax.set_xlabel("datapoint in sample")
            ax.set_ylabel("highgamma power envelope")

            ax.legend()
            ax.set_title("epoch " + str(e + 1))

        plt.suptitle("Original and recon signal for elec G" + str(key))

        fig.tight_layout()

        pdf_pages.savefig(fig)
        plt.close(fig)

    # Close the PDF file
    pdf_pages.close()