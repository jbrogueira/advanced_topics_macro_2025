# PhD Advanced Topics in Macro, 2025\*

<img src="NOVASBE-LOGO.png" alt="Nova SBE Logo" width="180">

## Course requirements

## Objective

This exercise aims to develop your skills in replicating quantitative research, understanding computational methods, and presenting results. You will replicate key findings from a selected paper, reproduce its main results, and document your methodology.

## Key Dates

- **Paper Proposal Due**: Monday 24th Nov.
  - Submit a one-paragraph description of your chosen paper and replication plan
  - Include confirmation of data availability
  
- **Final Submission Due**: Friday, 19th Dec.
  - Complete replication package (code + PDF report)

## Requirements

### 1. Paper Selection
Choose ONE paper (or instructor-assigned alternative) from:
- A recent paper from top a journal (e.g. AER, QJE, JPE, Econometrica)
- Papers involving quantitative macro models, or panel data methods are preferred
- The paper must have publicly available data or allow for data construction from standard sources. 

### 2. Code Deliverables

**Python Implementation Requirements:**
- Create well-documented Python code that replicates at least **two main tables/figures** from the paper
- Use appropriate libraries: `pandas`, `numpy`, `matplotlib`, `statsmodels`, or `linearmodels`
- **Special emphasis on JAX**: When applicable, leverage JAX for:
  - Automatic differentiation in solving/estimating models
  - GPU/TPU acceleration for computationally intensive tasks
  - Vectorized operations for efficiency gains
  - JIT compilation for performance optimization
- Code must be modular with clear function definitions
- Include detailed comments explaining each step of the methodology
- Provide a `requirements.txt` file listing all dependencies
- Code should run without errors when executed in sequence

**Code Structure:**
```
replication_project/
├── data/                  # Raw and processed data
├── code/                  # Python scripts (.py files)
├── results/               # Output tables and figures
├── requirements.txt       # List of all Python dependencies
└── README.md              # Brief guide to reproduce results
```

### 3. Written Report (PDF)

Submit a **5-7 page PDF** containing:

1. **Introduction** (0.5 pages): Brief summary of the original paper's research question and main findings

2. **Methodology** (1-2 pages): 
   - Describe the empirical/theoretical approach
   - Explain computational methods employed (highlight JAX usage if applicable)
   - Document any modifications made to the original methodology
   - Discuss data sources and any necessary adjustments

3. **Results** (2-3 pages):
   - Present replicated tables and figures
   - Compare your results with the original paper
   - Discuss any discrepancies and potential reasons
   - Report computational efficiency gains (if JAX was used)

4. **Challenges and Reflections** (0.5 page):
   - Technical difficulties encountered
   - Computational challenges and solutions
   - Lessons learned

### 4. Submission Guidelines

- **Due Date**: [To be announced]
- **Format**: 
  - One ZIP file containing code folder and PDF report
  - Name format: `LastName_FirstName_Replication.zip`
- **Code**: Must include a main script that executes the entire analysis
- **PDF**: Standard academic formatting (12pt font, 1-inch margins)

### 5. Evaluation Criteria

- **Accuracy** (30%): Closeness of replicated results to original findings
- **Code Quality** (30%): Documentation, structure, reproducibility, and computational efficiency
- **Written Presentation** (20%): Clarity of exposition and analysis
- **Critical Thinking** (20%): Insightful discussion of methodology, computational approaches, and discrepancies

## Resources

- **Python Resources**: QuantEcon lectures, JAX documentation, course notebooks.
- **JAX Tutorials**: https://jax.readthedocs.io/

## Academic Integrity

All submitted code must be your own work. While you are encouraged to use AI assistants (e.g., GitHub Copilot) as learning and debugging tools, you bear full responsibility for the correctness and integrity of your final submission. You may consult with classmates on technical issues, but final implementation must be independent. Properly cite any external code libraries or resources used beyond standard Python packages.


---

*Course: Advanced Topics in Macroeconomics, Nova School of Business and Economics*