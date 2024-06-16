import pandas as pd
from logger import AppLogger

logger = AppLogger(__name__).logger


class PromptPrepper:
    def __init__(self, system_prompt, benchmark_name, sample) -> None:
        self.system_prompt = system_prompt
        self.sample = sample
        self.benchmark_name = benchmark_name
        self.prepped_sample = []

    def prompt_prep_arc(self):
        choices = [f"{x+1}: {y}" for x, y in enumerate(self.sample["choices"]["text"])]
        choices = "\n".join(choices)
        message = """{question}\nПонудени одговори:\n{choices}\nСамо еден од одговорите е точен. Одговори само со бројот на точниот одговор."""
        message = message.format(question=self.sample["question"], choices=choices)
        return message

    def prompt_prep_winogrande(self):
        choices = f'1: {self.sample["option1"]}\n2: {self.sample["option2"]}'
        message = """Избери еден од понудените одговори за точно да го пополниш празното место во реченицата.\nРеченица:{question}\nПонудени одговори:\n{choices}\nОдговори само со бројот на точниот одговор."""
        message = message.format(question=self.sample["sentence"], choices=choices)
        return message

    def prompt_prep_copa(self):
        cause = "Ја бараме причината за премисата."
        effect = "Го бараме ефектот на  премисата."
        if self.sample["question"] == "cause":
            cause_effect = cause
        else:
            cause_effect = effect
        message = """Со оглед на следнава премиса: "{premise}". {cause_effect}. Која хипотеза има повеќе смисла? Хипотеза 1: "{hypothesis_1}". Хипотеза 2: "{hypothesis_2}". Одговори само со "1" или "2"."""
        message = message.format(
            premise=self.sample["premise"],
            cause_effect=cause_effect,
            hypothesis_1=self.sample["choice1"],
            hypothesis_2=self.sample["choice2"],
        )
        return message

    def sample_prep(self):
        system_prompt_content = "Разговор помеѓу љубопитен корисник и асистент со вештачка интелигенција. Асистентот дава корисни, детални и љубезни одговори на прашањата на корисникот. Асистентот е развиен од група македонски инженери и научници."
        system_turn = {"role": "system", "content": system_prompt_content}
        message_history = []
        if self.system_prompt:
            message_history.append(system_turn)

        if self.benchmark_name in ["arc_easy_mk", "arc_challenge_mk"]:
            message = self.prompt_prep_arc()
        elif self.benchmark_name == "COPA-MK":
            message = self.prompt_prep_copa()
        elif self.benchmark_name == "winogrande_mk":
            message = self.prompt_prep_winogrande()
        message_history.append({"role": "user", "content": message})

        self.prepped_sample = message_history


class ResultsParser:
    def __init__(self, benchmark_name, data, responses) -> None:
        self.benchmark_name = benchmark_name
        self.data = data
        self.responses = responses
        self.dataset = {"prompt": [], "true": [], "predicted": []}

    def results_prep_arc(self):
        answer_mapper = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

        for true, predicted in zip(self.data, self.responses):
            try:
                predicted = predicted.strip()
                predicted = int(predicted[0])
                self.dataset["predicted"].append(predicted)
                self.dataset["prompt"].append(
                    f'{true["question"]} -- {true["choices"]}'
                )
                try:
                    self.dataset["true"].append(int(true["answerKey"]))
                except:
                    self.dataset["true"].append(answer_mapper[true["answerKey"]])
            except:
                logger.warn(f"Issue with parsing response: {predicted}")
        df = pd.DataFrame(self.dataset)
        score = sum(df["true"] == df["predicted"]) / df.shape[0]
        score = round(score, 2)
        logger.info(f"Score: {score}")
        logger.info(f"Saving output file: {self.benchmark_name}.csv")
        df.to_csv(f"{self.benchmark_name}.csv", index=False)

    def results_prep_copa(self):
        for true, predicted in zip(self.data, self.responses):
            try:
                predicted = predicted.strip()
                predicted = int(predicted[0])
                self.dataset["predicted"].append(predicted)
                self.dataset["true"].append(int(true["label"]) + 1)
                self.dataset["prompt"].append(
                    f'{true["premise"]} -- {true["choice1"]}, {true["choice2"]}'
                )
            except:
                logger.warn(f"Issue with parsing response: {predicted}")
        df = pd.DataFrame(self.dataset)
        score = sum(df["true"] == df["predicted"]) / df.shape[0]
        score = round(score, 2)
        logger.info(f"Score: {score}")
        logger.info(f"Saving output file: {self.benchmark_name}.csv")
        df.to_csv(f"{self.benchmark_name}.csv", index=False)

    def results_prep_winogrande(self):
        for true, predicted in zip(self.data, self.responses):
            try:
                predicted = predicted.strip()
                predicted = int(predicted[0])
                self.dataset["predicted"].append(predicted)
                self.dataset["true"].append(int(true["answer"]))
                self.dataset["prompt"].append(
                    f'{true["sentence"]} -- {true["option1"]}, {true["option2"]}'
                )
            except:
                logger.warn(f"Issue with parsing response: {predicted}")
        df = pd.DataFrame(self.dataset)
        score = sum(df["true"] == df["predicted"]) / df.shape[0]
        score = round(score, 2)
        logger.info(f"Score: {score}")
        logger.info(f"Saving output file: {self.benchmark_name}.csv")
        df.to_csv(f"{self.benchmark_name}.csv", index=False)

    def prep_results(self):
        if self.benchmark_name in ["arc_easy_mk", "arc_challenge_mk"]:
            self.results_prep_arc()
        elif self.benchmark_name == "COPA-MK":
            self.results_prep_copa()
        elif self.benchmark_name == "winogrande_mk":
            self.results_prep_winogrande()
