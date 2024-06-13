import shutil
import os
import datetime
import uuid
import re
import json


class logger(object):
    def __init__(self, methods=["generate", "template search", "beam search"]):
        self.methods = methods

    def __call__(self, generating):
        def wrapper(*args, **kwargs):
            parameter = args[0]
            self.methods = []
            if parameter.test:
                parameter.generate = True
                parameter.template_search = True
                parameter.beam_search = True
            if parameter.generate:
                self.methods.append("generate")
            if parameter.generate_repeated_penalties:
                self.methods.append("generate")
            if parameter.template_search:
                self.methods.append("template search")
            if parameter.beam_search:
                self.methods.append("beam search")

            resultlog = ResultLog(output=parameter.model_path, model_name=parameter.model, epoch=parameter.test_epoch,
                                  generates=self.methods)
            parameter.log = resultlog

            generating(*args, **kwargs)

        return wrapper


class ResultLog():
    def __init__(self, output, model_name, epoch=None, generates=["generate"]):
        if not epoch:
            epoch = datetime.datetime.now()
            epoch = re.sub("( |:|\.)", "-", str(epoch))
        path = f'result/{model_name}'
        if not os.path.exists(path):
            os.mkdir(path)
        path = f'result/{model_name}/{output}'
        if not os.path.exists(path):
            os.mkdir(path)
        path = f'result/{model_name}/{output}/epoch{epoch}'
        if not os.path.exists(path):
            os.mkdir(path)
        _cache_path = f'result/{model_name}/{output}/epoch{epoch}/cache/'
        if not os.path.exists(_cache_path):
            os.mkdir(_cache_path)
        else:
            shutil.rmtree(_cache_path)
            os.mkdir(_cache_path)
        for generating in generates:

            _cache_path = f'result/{model_name}/{output}/epoch{epoch}/cache/{generating}'
            if not os.path.exists(_cache_path):
                os.mkdir(_cache_path)
        # self.epoch = epoch
        self.path = path
        self.generates = generates

    def write_cache(self, img_ids, template_ids, decoded_preds, generate="generate", cache_key_top_results=None):

        temp_dict = {}

        if len(img_ids) == len(decoded_preds):
            for i in range(len(img_ids)):
                uuid4 = uuid.uuid4()
                temp_dict[f"{img_ids[i]}_{template_ids[i]}_{uuid4}"] = decoded_preds[i]
                if cache_key_top_results:
                    temp_dict[f"{img_ids[0]}_{template_ids[i]}"] = cache_key_top_results
        elif len(img_ids) == 1:
            for i in range(len(decoded_preds)):
                uuid4 = uuid.uuid4()
                temp_dict[f"{img_ids[0]}_{template_ids[0]}_{uuid4}"] = decoded_preds[i]
            # if cache_key_top_results:
            #     temp_dict[f"{img_ids[0]}"] = cache_key_top_results
        else:
            raise Exception("img ids and  decoded preds Not Match")
        uuid4 = uuid.uuid4()
        with open(f"{self.path}/cache/{generate}/{uuid4}.json", "w", encoding="utf-8") as f:
            json.dump(temp_dict, f, ensure_ascii=False)
            f.close()

    # todo a interface to cache log write json file
    def write(self):

        ttime = datetime.datetime.now()
        ttime = re.sub("( |:|\.)", "-", str(ttime))
        for generate in self.generates:
            cache_path = f"{self.path}/cache/{generate}"
            temp_dict = {}
            for cache_file in os.listdir(cache_path):
                with open(f"{cache_path}/{cache_file}", "r", encoding="utf-8") as f:
                    t = json.load(f)
                    temp_dict.update(t)
            with open(f"{self.path}/{generate}_{ttime}.json", "w", encoding="utf-8") as f:
                json.dump(temp_dict, f)


if __name__ == '__main__':
    img_ids = [4]
    decoded_preds = ["123456"]

    resultlog = ResultLog("output-1", "test")
    resultlog.write_cache(img_ids, decoded_preds)
    resultlog.write_cache(img_ids, decoded_preds)
    resultlog.write_cache(img_ids, decoded_preds)
    resultlog.write_cache(img_ids, decoded_preds)
    resultlog.write_cache(img_ids, decoded_preds)
    resultlog.write()
