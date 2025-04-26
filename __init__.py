import comfy.samplers
import comfy.sample
from comfy.utils import ProgressBar
from .utils import expand_mask, FONTS_DIR, parse_string_to_list
import logging
import folder_paths
import latent_preview
import torch
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
from nodes import MAX_RESOLUTION
from node_helpers import conditioning_set_values

# https://github.com/cubiq/ComfyUI_essentials/blob/main/sampling.py


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
        from comfy_extras.nodes_latent import LatentBatch
        from nodes import LoraLoader

        is_schnell = model.model.model_type == comfy.model_base.ModelType.FLOW

        noise = [random.randint(0, 999999) if -1 == seed else int(seed)]
        seed = str(seed)
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

class StyleModelApplyHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                'clip_name': (folder_paths.get_filename_list('clip_vision'),),
                "image": ("IMAGE",),
                "crop": (["center", "none"],),
                "conditioning": ("CONDITIONING",),  
                "style_model_name": (folder_paths.get_filename_list("style_models"), ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "strength_type": (["multiply", "attn_bias"], ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "AKNodes/conditioning"

    def execute(self, clip_name, image, crop, conditioning, style_model_name, strength, strength_type):
        import torch
        
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        if clip_vision is None:
            raise RuntimeError("ERROR: clip vision file is invalid and does not contain a valid vision model.")

        crop_image = True
        if crop != "center":
            crop_image = False
        clip_vision_output = clip_vision.encode_image(image, crop=crop_image)

        style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model_name)
        style_model = comfy.sd.load_style_model(style_model_path)

        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            # even if the strength is 1.0 (i.e, no change), if there's already a mask, we have to add to it
            if "attention_mask" in keys or (strength_type == "attn_bias" and strength != 1.0):
                # math.log raises an error if the argument is zero
                # torch.log returns -inf, which is what we want
                attn_bias = torch.log(torch.Tensor([strength if strength_type == "attn_bias" else 1.0]))
                # get the size of the mask image
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]
                # grab the existing mask
                mask = keys.get("attention_mask", None)
                # create a default mask if it doesn't exist
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                # convert the mask dtype, because it might be boolean
                # we want it to be interpreted as a bias
                if mask.dtype == torch.bool:
                    # log(True) = log(1) = 0
                    # log(False) = log(0) = -inf
                    mask = torch.log(mask.to(dtype=torch.float16))
                # now we make the mask bigger to add space for our new tokens
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                # copy over the old mask, in quandrants
                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                # now fill in the attention bias to our redux tokens
                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        return (c_out,)

class FluxSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,}),
                    "model": ("MODEL",),
                    "conditioning": ("CONDITIONING", ),
                    "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                    "latent_image": ("LATENT", ),
                    "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                    "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01}),
                    "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "execute"

    CATEGORY = "AKNodes/sampling"

    def execute(self, noise_seed, model, conditioning, sampler_name, latent_image, scheduler, steps, guidance, max_shift, base_shift, denoise):
        is_schnell = model.model.model_type == comfy.model_base.ModelType.FLOW
        noise = Noise_RandomNoise(noise_seed)
        
        modelsamplingflux = ModelSamplingFlux() if not is_schnell else ModelSamplingAuraFlow()
        width = latent_image["samples"].shape[3]*8
        height = latent_image["samples"].shape[2]*8

        if is_schnell:
            work_model = modelsamplingflux.patch_aura(model, base_shift)[0]
        else:
            work_model = modelsamplingflux.patch(model, max_shift, base_shift, width, height)[0]

        cond = conditioning_set_values(conditioning, {"guidance": guidance})

        basicguider = BasicGuider()
        guider = basicguider.get_guider(work_model, cond)[0]

        sampler = comfy.samplers.sampler_object(sampler_name)

        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(steps/denoise)

        sigmas = comfy.samplers.calculate_sigmas(work_model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]

        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

class FluxSamplerWithGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,}),                 
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                "guider": ("GUIDER", ),
                "latent_image": ("LATENT", ),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "execute"

    CATEGORY = "AKNodes/sampling"

    def execute(self, noise_seed, sampler_name, latent_image, scheduler, steps,
                max_shift, base_shift, denoise, guider=None):
        is_schnell = guider.model_patcher.model.model_type == comfy.model_base.ModelType.FLOW
        noise = Noise_RandomNoise(noise_seed)
        
        modelsamplingflux = ModelSamplingFlux() if not is_schnell else ModelSamplingAuraFlow()
        width = latent_image["samples"].shape[3]*8
        height = latent_image["samples"].shape[2]*8

        if is_schnell:
            work_model = modelsamplingflux.patch_aura(guider.model_patcher, base_shift)[0]
        else:
            work_model = modelsamplingflux.patch(guider.model_patcher, max_shift, base_shift, width, height)[0]

        sampler = comfy.samplers.sampler_object(sampler_name)

        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(steps/denoise)

        sigmas = comfy.samplers.calculate_sigmas(work_model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]

        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

class EmptyLatentFromImageDimensions:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (["704x1408 (0.5)","704x1344 (0.52)","768x1344 (0.57)","768x1280 (0.6)","832x1216 (0.68)","832x1152 (0.72)","896x1152 (0.78)","896x1088 (0.82)","960x1088 (0.88)","960x1024 (0.94)","1024x1024 (1.0)","1024x960 (1.07)","1088x960 (1.13)","1088x896 (1.21)","1152x896 (1.29)","1152x832 (1.38)","1216x832 (1.46)","1280x768 (1.67)","1344x768 (1.75)","1344x704 (1.91)","1408x704 (2.0)","1472x704 (2.09)","1536x640 (2.4)","1600x640 (2.5)","1664x576 (2.89)","1728x576 (3.0)",], {"default": "1024x1024 (1.0)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT","INT","INT",)
    RETURN_NAMES = ("LATENT","width","height",)
    FUNCTION = "execute"
    CATEGORY = "AKNodes/Latent"
    DESCRIPTION = """
根据图片尺寸创建Latent,返回Latent和宽高

    """

    def execute(self, resolution, batch_size, width_override=0, height_override=0, image=None):
        if( image is not None):
            _, raw_H, raw_W, _ = image.shape

            width = raw_W
            height = raw_H
        else:
            width, height = resolution.split(" ")[0].split("x")
            width = width_override if width_override > 0 else int(width)
            height = height_override if height_override > 0 else int(height)

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return {"ui": {
            "text": [f"{width}x{height}"]}, 
            "result": ({"samples":latent}, width, height,)
        }

NODE_CLASS_MAPPINGS = {
    # "FluxSimpleSamplerParams": FluxSimpleSamplerParams,
    "FluxSampler": FluxSampler,
    "FluxSamplerWithGuider": FluxSamplerWithGuider,
    "StyleModelEfficiency": StyleModelApplyHelper,
    "EmptyLatentFromImageDimensions": EmptyLatentFromImageDimensions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "FluxSimpleSamplerParams": "Flux简单采样",
    "FluxSampler": "Flux简易采样器",
    "FluxSamplerWithGuider": "Flux带引导采样器",
    "StyleModelEfficiency": "风格模型应用助手",
    "EmptyLatentFromImageDimensions": "从图片大小创建Latent",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./js"