{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMMXV3y4l1TVHYl8gLun+mz",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "ir",
      "display_name": "R"
    },
    "language_info": {
      "name": "R"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/venkatanadikatla/pytorch/blob/main/Hierarchical%20Normal%20Model%20%26%20Gibbs%20Sampling.R\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Data Overview:**\n",
        "The dataset consists of annual death counts from five causes for the age group 25–34 over five years (2018–2022). These causes are:\n",
        "\n",
        "Accidents, Homicide, Suicide, Heart Disease, Malignant Neoplasms\n",
        "The goal is to model these death counts using three approaches:\n",
        "\n",
        "\n",
        "*   Separate Gaussian model\n",
        "\n",
        "*   Pooled Gaussian model\n",
        "*   Hierarchical Gaussian model\n",
        "\n",
        "**Why Gibbs Sampling:**\n",
        "Gibbs sampling is a special case of the Metropolis-Hastings algorithm used for Bayesian inference, particularly when the full conditional distributions of the variables in a model can be easily sampled.\n",
        "\n",
        "**a. Separate Model:** In a separate model, we treat each group or data source independently. When using Gibbs sampling here, each group's parameters are sampled independently from their respective conditional distributions.\n",
        "\n",
        "\n",
        "\n",
        "*   **Specific to the model:**  Sample the parameters for each cause of death independently. Each death cause has its own parameters, which could lead to high variability in small samples.\n",
        "\n",
        "\n",
        "**b. Pooled Model:** In a pooled model, the data from different groups are combined (or \"pooled\") and treated as if they came from a single distribution. Gibbs sampling in this model would sample the parameters from a single conditional distribution across all data. This reduces variance but may miss group-specific effects, assuming all groups are homogenous.\n",
        "\n",
        "*   **Specific to the model:**  Pooled Model (Gibbs): Sample from a common parameter for all causes, assuming homogeneity.\n",
        "\n",
        "**c. Hierarchical Model** (or Partially Pooled Model): In hierarchical models, group-level parameters (for individual groups) are treated as random variables drawn from a higher-level (hyperparameter) distribution. Gibbs sampling here involves sampling both the group-level parameters and the hyperparameters in a conditional step-by-step process. This approach balances between separate and pooled models, allowing group-specific variability while sharing information across groups.\n",
        "\n",
        "*   **Specific to the model:** Sample group-specific parameters and hyperparameters, allowing for shared information while maintaining some independence between causes.\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "htIVRYa-bLJz"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "2QWzYA2OZjZ0",
        "outputId": "d99c05ff-53af-4859-cc6b-64e8f02b936c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Reading package lists... Done\n",
            "Building dependency tree... Done\n",
            "Reading state information... Done\n",
            "r-base is already the newest version (4.4.1-3.2204.0).\n",
            "0 upgraded, 0 newly installed, 0 to remove and 49 not upgraded.\n"
          ]
        }
      ],
      "source": [
        "!apt-get install -y r-base\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Install the IRkernel package for R support\n",
        "!R -e \"IRkernel::installspec(user = FALSE)\"\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "5xDHGV1XlkgV",
        "outputId": "492ae828-9225-4fce-ce6c-ac087048a779"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "R version 4.4.1 (2024-06-14) -- \"Race for Your Life\"\n",
            "Copyright (C) 2024 The R Foundation for Statistical Computing\n",
            "Platform: x86_64-pc-linux-gnu\n",
            "\n",
            "R is free software and comes with ABSOLUTELY NO WARRANTY.\n",
            "You are welcome to redistribute it under certain conditions.\n",
            "Type 'license()' or 'licence()' for distribution details.\n",
            "\n",
            "  Natural language support but running in an English locale\n",
            "\n",
            "R is a collaborative project with many contributors.\n",
            "Type 'contributors()' for more information and\n",
            "'citation()' on how to cite R or R packages in publications.\n",
            "\n",
            "Type 'demo()' for some demos, 'help()' for on-line help, or\n",
            "'help.start()' for an HTML browser interface to help.\n",
            "Type 'q()' to quit R.\n",
            "\n",
            "> IRkernel::installspec(user = FALSE)\n",
            "\u001b[?25h> \n",
            "> \n",
            "\u001b[?25h"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Install dependencies and RStan\n",
        "install.packages(\"rstan\", repos = \"https://cloud.r-project.org/\", dependencies=TRUE)\n",
        "\n",
        "# Load the library\n",
        "library(rstan)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "ru3LMKIylk8I",
        "outputId": "ad76cf0c-cff2-4e98-efbd-1f36f073f7e6"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Installing package into ‘/usr/local/lib/R/site-library’\n",
            "(as ‘lib’ is unspecified)\n",
            "\n",
            "also installing the dependencies ‘numDeriv’, ‘lazyeval’, ‘abind’, ‘tensorA’, ‘distributional’, ‘crosstalk’, ‘zoo’, ‘plyr’, ‘igraph’, ‘checkmate’, ‘matrixStats’, ‘posterior’, ‘colourpicker’, ‘DT’, ‘dygraphs’, ‘gtools’, ‘markdown’, ‘reshape2’, ‘shinyjs’, ‘shinythemes’, ‘threejs’, ‘xts’, ‘ggridges’, ‘StanHeaders’, ‘inline’, ‘gridExtra’, ‘RcppParallel’, ‘loo’, ‘QuickJSR’, ‘RcppEigen’, ‘BH’, ‘shinystan’, ‘bayesplot’, ‘rstantools’, ‘coda’, ‘V8’\n",
            "\n",
            "\n",
            "Loading required package: StanHeaders\n",
            "\n",
            "\n",
            "rstan version 2.32.6 (Stan version 2.32.2)\n",
            "\n",
            "\n",
            "For execution on a local, multicore CPU with excess RAM we recommend calling\n",
            "options(mc.cores = parallel::detectCores()).\n",
            "To avoid recompilation of unchanged Stan programs, we recommend calling\n",
            "rstan_options(auto_write = TRUE)\n",
            "For within-chain threading using `reduce_sum()` or `map_rect()` Stan functions,\n",
            "change `threads_per_chain` option:\n",
            "rstan_options(threads_per_chain = 1)\n",
            "\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**General Setup:**"
      ],
      "metadata": {
        "id": "h6mZm_uvxnz4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load necessary libraries\n",
        "library(rstan)\n",
        "\n",
        "# Prepare the data\n",
        "data_list <- list(\n",
        "  N = 5, # 5 years of data\n",
        "  C = 5, # 5 causes of death\n",
        "  y = matrix(c(33058, 34452, 31315, 24516, 24614,\n",
        "               6712, 7571, 31315, 5341, 5234,\n",
        "               8663, 8862, 8454, 8059, 8020,\n",
        "               3789, 4155, 3984, 3495, 3561,\n",
        "               3641, 3615, 3573, 3577, 3684), ncol=5, byrow=TRUE)\n",
        ")\n"
      ],
      "metadata": {
        "id": "VkSRFaVwqAkl"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**1. Separate Gaussian Model:** Each cause has its own mean and shared variance"
      ],
      "metadata": {
        "id": "PiFHsp-eycE2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the Stan model as a string\n",
        "stan_code_separate <- \"\n",
        "data {\n",
        "  int<lower=0> N; // number of observations (years)\n",
        "  int<lower=0> C; // number of causes\n",
        "  matrix[C, N] y; // observed death counts\n",
        "}\n",
        "\n",
        "parameters {\n",
        "  vector[C] mu;  // separate means for each cause\n",
        "  real<lower=0> sigma; // shared standard deviation\n",
        "}\n",
        "\n",
        "model {\n",
        "  for (i in 1:C) {\n",
        "    y[i] ~ normal(mu[i], sigma);\n",
        "  }\n",
        "  mu ~ normal(0, 100); // weakly informative prior\n",
        "  sigma ~ cauchy(0, 2.5); // weakly informative prior\n",
        "}\n",
        "\"\n"
      ],
      "metadata": {
        "id": "bgDonGyJxwpB"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fit_separate <- stan(file = 'separate_gaussian_model.stan', data = data_list, iter = 2000, chains = 4)\n",
        "\n",
        "# Check results: posterior distribution of means\n",
        "print(fit_separate, pars = c(\"mu\", \"sigma\"))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 591
        },
        "id": "96A25CMQyvUU",
        "outputId": "e4efa0a1-abe8-4017-95c8-3b5aba222e36"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Warning message in normalizePath(file):\n",
            "“path[1]=\"separate_gaussian_model.stan\": No such file or directory”\n",
            "Warning message in file(fname, \"rt\"):\n",
            "“cannot open file 'separate_gaussian_model.stan': No such file or directory”\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Error in file(fname, \"rt\") : cannot open the connection\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "ERROR",
          "evalue": "Error in get_model_strcode(file, model_code): cannot open model file \"separate_gaussian_model.stan\"\n",
          "traceback": [
            "Error in get_model_strcode(file, model_code): cannot open model file \"separate_gaussian_model.stan\"\nTraceback:\n",
            "1. stan_model(file, model_name = model_name, model_code = model_code, \n .     stanc_ret = NULL, boost_lib = boost_lib, eigen_lib = eigen_lib, \n .     save_dso = save_dso, verbose = verbose)",
            "2. stanc(file = file, model_code = model_code, model_name = model_name, \n .     verbose = verbose, obfuscate_model_name = obfuscate_model_name, \n .     allow_undefined = allow_undefined, allow_optimizations = allow_optimizations, \n .     standalone_functions = standalone_functions, use_opencl = use_opencl, \n .     warn_pedantic = warn_pedantic, warn_uninitialized = warn_uninitialized, \n .     isystem = isystem)",
            "3. stanc_process(file = file, model_code = model_code, model_name = model_name, \n .     auto_format = FALSE, isystem = isystem)",
            "4. get_model_strcode(file, model_code)",
            "5. stop(paste(\"cannot open model file \\\"\", fname, \"\\\"\", sep = \"\"))",
            "6. .handleSimpleError(function (cnd) \n . {\n .     watcher$capture_plot_and_output()\n .     cnd <- sanitize_call(cnd)\n .     watcher$push(cnd)\n .     switch(on_error, continue = invokeRestart(\"eval_continue\"), \n .         stop = invokeRestart(\"eval_stop\"), error = invokeRestart(\"eval_error\", \n .             cnd))\n . }, \"cannot open model file \\\"separate_gaussian_model.stan\\\"\", \n .     base::quote(get_model_strcode(file, model_code)))"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "zUga3Mr00Jcm"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}