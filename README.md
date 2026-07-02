Repository that describes and reproduce the experiments presented in the paper "A Retrieval-Augmented LLM Pipeline for Generating
Cognitive Conceptualization Knowledge Graphs"

A basic structure and description of the repository is the following: 
- Vignettes files contains the input description of the patients, used as input of the pipeline.
- extract.py contains the main pipelines
- eval_x files contains the specifics evaluation of the method ecplicited in the name
- eval_pipeline is the complete method proposed
- llm_as_judge file perform the qualitative assessment, need and OPENAI_KEY
- The output already produced are present in the outputs_x folders divided by sit1 and sit2 (depending on the file used vignettes1 or vignettes2), the best run (the one produced with the proposed method) are outputs_pipeline_sit1 e output pipeline_sit2
- The proposed pipeline requires an OPENAI_KEY. 
- The train_x files fine tune the roberta models using the core belief dataset described in the paper 
