# Statbot.swiss

The repository contains all datasets and evaluations for the statbot.swiss benchmark.

## Install

- `conda create -n statbot python=3.11.4`

- `conda activate statbot`

- `pip install -r requirements.txt`

## Usage 

### To run langchain with ChatGPT:

1. provide openai-api-key in `src/config.json` file. 

2. reproduce results based on _gpt-3.5-turbo-16k_ run the command below.

3. For random shot selection, choose `random` and for the selection based on similarity choose `similarity` for `--shot-selection-strategy`.

- Command to run:

  `python src/main.py  --shot-selection-strategy <random|similarity>`

## Aknowlegement
This work is the output of the [INODE4StatBot.swiss](https://www.zhaw.ch/en/research/research-database/project-detailview/projektid/5959/) project, funded by the **Swiss Federal Statistical Office**. <br>
It is a collaboration between the **Institute of Computer Science** at **ZHAW**, the **Competence Center for Data Science of the Federal Statistical Office**, the **Cantonal Statistical Office Zurich**, and the **Swiss Data Science Center**.

## License
This work is licensed under the MIT license.
Third-party software and data are subject to their respective licenses. <br>

## Citation
Please cite the paper as follows:

```
@article{statbot2024,
  title={StatBot. Swiss: Bilingual Open Data Exploration in Natural Language},
  author={Nooralahzadeh, Farhad and Zhang, Yi and Smith, Ellery and Maennel, Sabine and Matthey-Doret, Cyril and de Fondville, Rapha{\"e}l and Stockinger, Kurt},
  journal={arXiv preprint arXiv:2406.03170},
  year={2024}
}
```