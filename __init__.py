import comfy.samplers
import comfy.sample
from comfy.utils import ProgressBar
from .utils import expand_mask, FONTS_DIR, parse_string_to_list
import logging

# https://github.com/cubiq/ComfyUI_essentials/blob/main/sampling.py

MAX_SEED_NUM = 1125899906842624
class FluxSimpleSamplerParams:
    def __init__(self):
        self.loraloader = None
        self.lora = (None, None)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "conditioning": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),

                    "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple" }),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                    "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01}),
                    "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
                "optional": {
                    "loras": ("LORA_PARAMS",),
                }}

    RETURN_TYPES = ("LATENT","SAMPLER_PARAMS")
    RETURN_NAMES = ("latent", "params")
    FUNCTION = "execute"
    CATEGORY = "AKNodes/sampling"

    def execute(self, model, conditioning, latent_image, seed, sampler, scheduler, steps, guidance, max_shift, base_shift, denoise, loras=None):
        import random
        import time
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
        from comfy_extras.nodes_latent import LatentBatch
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
        from node_helpers import conditioning_set_values
        from nodes import LoraLoader

        is_schnell = model.model.model_type == comfy.model_base.ModelType.FLOW

        seed = str(seed)
        noise = seed.replace("\n", ",").split(",")
        noise = [random.randint(0, 999999) if "?" in n else int(n) for n in noise]
        sampler = [sampler]
        scheduler = [scheduler]
        if not noise:
            noise = [random.randint(0, 999999)]

        steps = str(steps)
        steps = parse_string_to_list(steps)

        denoise = str(denoise)
        denoise = parse_string_to_list(denoise)

        guidance = str(guidance)
        guidance = parse_string_to_list(guidance)

        base_shift = str(base_shift)
        max_shift = str(max_shift)
        if not is_schnell:
            max_shift = "1.15" if max_shift == "" else max_shift
            base_shift = "0.5" if base_shift == "" else base_shift
        else:
            max_shift = "0"
            base_shift = "1.0" if base_shift == "" else base_shift

        max_shift = parse_string_to_list(max_shift)
        base_shift = parse_string_to_list(base_shift)

        cond_text = None
        if isinstance(conditioning, dict) and "encoded" in conditioning:
            cond_text = conditioning["text"]
            cond_encoded = conditioning["encoded"]
        else:
            cond_encoded = [conditioning]

        out_latent = None
        out_params = []

        basicschedueler = BasicScheduler()
        basicguider = BasicGuider()
        samplercustomadvanced = SamplerCustomAdvanced()
        latentbatch = LatentBatch()
        modelsamplingflux = ModelSamplingFlux() if not is_schnell else ModelSamplingAuraFlow()
        width = latent_image["samples"].shape[3]*8
        height = latent_image["samples"].shape[2]*8

        lora_strength_len = 1
        if loras:
            lora_model = loras["loras"]
            lora_strength = loras["strengths"]
            lora_strength_len = sum(len(i) for i in lora_strength)

            if self.loraloader is None:
                self.loraloader = LoraLoader()

        # count total number of samples
        total_samples = len(cond_encoded) * len(noise) * len(max_shift) * len(base_shift) * len(guidance) * len(sampler) * len(scheduler) * len(steps) * len(denoise) * lora_strength_len
        current_sample = 0
        if total_samples > 1:
            pbar = ProgressBar(total_samples)

        lora_strength_len = 1
        if loras:
            lora_strength_len = len(lora_strength[0])

        for los in range(lora_strength_len):
            if loras:
                patched_model = self.loraloader.load_lora(model, None, lora_model[0], lora_strength[0][los], 0)[0]
            else:
                patched_model = model

            for i in range(len(cond_encoded)):
                conditioning = cond_encoded[i]
                ct = cond_text[i] if cond_text else None
                for n in noise:
                    randnoise = Noise_RandomNoise(n)
                    for ms in max_shift:
                        for bs in base_shift:
                            if is_schnell:
                                work_model = modelsamplingflux.patch_aura(patched_model, bs)[0]
                            else:
                                work_model = modelsamplingflux.patch(patched_model, ms, bs, width, height)[0]
                            for g in guidance:
                                cond = conditioning_set_values(conditioning, {"guidance": g})
                                guider = basicguider.get_guider(work_model, cond)[0]
                                for s in sampler:
                                    samplerobj = comfy.samplers.sampler_object(s)
                                    for sc in scheduler:
                                        for st in steps:
                                            for d in denoise:
                                                sigmas = basicschedueler.get_sigmas(work_model, sc, st, d)[0]
                                                current_sample += 1
                                                log = f"Sampling {current_sample}/{total_samples} with seed {n}, sampler {s}, scheduler {sc}, steps {st}, guidance {g}, max_shift {ms}, base_shift {bs}, denoise {d}"
                                                lora_name = None
                                                lora_str = 0
                                                if loras:
                                                    lora_name = lora_model[0]
                                                    lora_str = lora_strength[0][los]
                                                    log += f", lora {lora_name}, lora_strength {lora_str}"
                                                logging.info(log)
                                                start_time = time.time()
                                                latent = samplercustomadvanced.sample(randnoise, guider, samplerobj, sigmas, latent_image)[1]
                                                elapsed_time = time.time() - start_time
                                                out_params.append({"time": elapsed_time,
                                                                "seed": n,
                                                                "width": width,
                                                                "height": height,
                                                                "sampler": s,
                                                                "scheduler": sc,
                                                                "steps": st,
                                                                "guidance": g,
                                                                "max_shift": ms,
                                                                "base_shift": bs,
                                                                "denoise": d,
                                                                "prompt": ct,
                                                                "lora": lora_name,
                                                                "lora_strength": lora_str})

                                                if out_latent is None:
                                                    out_latent = latent
                                                else:
                                                    out_latent = latentbatch.batch(out_latent, latent)[0]
                                                if total_samples > 1:
                                                    pbar.update(1)

        return (out_latent, out_params)


NODE_CLASS_MAPPINGS = {
    "FluxSimpleSamplerParams": FluxSimpleSamplerParams
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxSimpleSamplerParams": "Flux简单采样"
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]