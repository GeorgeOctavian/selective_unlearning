import torch


class TaskVector():
    def __init__(self, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            with torch.no_grad():
                self.vector = vector
        else:
            raise Exception("weights dictionary is empty!")
        
    def __add__(self, other, scaling_coef = 1):
        if other is None:
            raise Exception("weights in added vector are None !")
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + scaling_coef * other.vector[key]
        return TaskVector(vector=new_vector)

    # def __radd__(self, other):
    #     if other is None or isinstance(other, int):
    #         return self
    #     return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    # def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
    #     """Apply a task vector to a pretrained model."""
    #     with torch.no_grad():
    #         pretrained_model = torch.load(pretrained_checkpoint)
    #         new_state_dict = {}
    #         pretrained_state_dict = pretrained_model.state_dict()
    #         for key in pretrained_state_dict:
    #             if key not in self.vector:
    #                 print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
    #                 continue
    #             new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
    #     pretrained_model.load_state_dict(new_state_dict, strict=False)
    #     return pretrained_model

