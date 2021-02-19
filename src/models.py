class PassThroughReWriter():
    def __init__(self):
        "simply use the raw query."
        pass
    def inference(self, samples):
        for sample_obj in samples:
            sample_obj["re-write"] = sample_obj['all_raw'][-1]
        return samples
    
class OracleReWriter():
    def __init__(self):
        "simply use the raw query."
        pass
    def inference(self, samples):
        for sample_obj in samples:
            sample_obj["re-write"] = sample_obj['all_man'][-1]
            sample_obj["raw query"] = sample_obj['all_raw'][-1]
        return samples