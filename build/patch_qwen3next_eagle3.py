#!/usr/bin/env python3
"""Patch: Add SupportsEagle3 interface to Qwen3NextForCausalLM.

Problem: Qwen3-Next doesn't implement the EAGLE3 interface, so
`--speculative-config '{"method":"eagle3",...}'` fails with:
  RuntimeError: Model does not support EAGLE3 interface but
  aux_hidden_state_outputs was requested

Fix: Add SupportsEagle3 to the class MRO, implement the 3 required methods,
and modify the Model's forward() to collect auxiliary hidden states.
"""

import os
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"


def apply():
    if not os.path.exists(TARGET):
        print(f"SKIP: {TARGET} not found")
        sys.exit(0)

    with open(TARGET) as f:
        content = f.read()

    if "SupportsEagle3" in content:
        print("SKIP: EAGLE3 patch already applied")
        return

    changes = 0

    # Fix 1: Add SupportsEagle3 to imports (multi-line import block)
    old_import = "    SupportsLoRA,\n    SupportsPP,\n)"
    new_import = "    SupportsEagle3,\n    SupportsLoRA,\n    SupportsPP,\n)"
    if "SupportsEagle3" not in content and old_import in content:
        content = content.replace(old_import, new_import, 1)
        changes += 1
        print("  [1] Added SupportsEagle3 import")
    elif "SupportsEagle3" in content:
        print("  [1] SupportsEagle3 import already present")
    else:
        print("  [1] Import pattern not found")

    # Fix 2: Add SupportsEagle3 to class MRO
    old_class = """\
class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):"""
    new_class = """\
class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle3,
    QwenNextMixtureOfExperts,
    IsHybrid,
):"""
    if old_class in content:
        content = content.replace(old_class, new_class)
        changes += 1
        print("  [2] Added SupportsEagle3 to class MRO")
    else:
        print("  [2] Class MRO pattern not found")

    # Fix 3: Add aux_hidden_state_layers to Qwen3NextModel.__init__
    old_model_init = "        if get_pp_group().is_last_rank:\n            self.norm = Qwen3NextRMSNorm"
    new_model_init = "        # EAGLE3: track layers for auxiliary hidden state outputs\n        self.aux_hidden_state_layers: tuple[int, ...] = ()\n\n        if get_pp_group().is_last_rank:\n            self.norm = Qwen3NextRMSNorm"
    if old_model_init in content:
        content = content.replace(old_model_init, new_model_init)
        changes += 1
        print("  [3] Added aux_hidden_state_layers to Qwen3NextModel")
    else:
        print("  [3] Model init pattern not found")

    # Fix 4: Modify Qwen3NextModel.forward() to collect aux hidden states
    old_model_forward = """\
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states"""
    new_model_forward = """\
        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            # EAGLE3: collect auxiliary hidden states
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_state = (
                    hidden_states + residual if residual is not None
                    else hidden_states
                )
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states"""
    if old_model_forward in content:
        content = content.replace(old_model_forward, new_model_forward)
        changes += 1
        print("  [4] Modified Model.forward() for aux hidden states")
    else:
        print("  [4] Model forward pattern not found")

    # Fix 5: Modify Qwen3NextForCausalLM.forward() to pass through aux hidden states
    old_causal_forward = """\
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states"""
    new_causal_forward = """\
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)"""
    if old_causal_forward in content:
        content = content.replace(old_causal_forward, new_causal_forward)
        changes += 1
        print("  [5] Added EAGLE3 methods to ForCausalLM")
    else:
        print("  [5] CausalLM forward pattern not found")

    if changes == 0:
        print("WARNING: No changes made")
        sys.exit(1)

    with open(TARGET, "w") as f:
        f.write(content)

    print(f"OK: {changes} EAGLE3 interface changes applied to {TARGET}")


if __name__ == "__main__":
    apply()
