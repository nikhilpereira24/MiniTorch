class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        training (bool): Whether the module is in training mode or evaluation mode

    # """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self):
        "Return the direct child modules of this module."
        return self.__dict__["_modules"].values()

    def getModules(self):
        "Return the direct child modules of this module."
        return self.__dict__["_modules"].items()

    def getParameters(self):
        "Return the direct child parameters of this module."
        return self.__dict__["_parameters"].items()

    def isChild(self):
        "Checks if the module is a child module or not based on the length of the dict"
        return len(self.getModules()) > 0

    def updateModulesParameters(self, runningDict, module_key):
        "Helper function to update the Modules Parameter Dictionary"
        if runningDict is None:  # checks if the running dictionary is empty
            return None
        else:
            for (
                pkey,
                pvalue,
            ) in (
                self.getParameters()
            ):  # for the parameters key, values in the modules parameter dictionary
                if (
                    module_key
                ):  # if there is a new module_key add that to the parameter key
                    runningDict[module_key + "." + pkey] = pvalue
                else:
                    runningDict[
                        pkey
                    ] = pvalue  # store the original parameters_key, parameters_value pairs
            return runningDict

    def train(self):
        "Set the mode of this module and all descendent modules to `train`."
        self.training = True  # set the modules value of train to true
        descendants = self.modules()  # create a dictionary of modules descendants
        for x in descendants:  # train all children
            x.train()

    def eval(self):
        self.training = False  # set the modules value of training to false
        descendants = self.modules()  # get all modules descendants
        for x in descendants:  # set all children to eval
            x.eval()

    def named_parameters(self):
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            list of pairs: Contains the name and :class:`Parameter` of each ancestor parameter.
        """
        parameters = {}
        for k, v in self._parameters.items():
            parameters[k] = v
        for mod_name, m in self._modules.items():
            for k, v in m.named_parameters():
                parameters[f"{mod_name}.{k}"] = v

        return parameters.items()
        # convert dict into list of tuples
        # return list_parameters
        # def helperRecursive(module, runningDict=None, parent_key=None):

        #     if runningDict is None:  # checks if the running dictionary is empty
        #         runningDict = {}  # creates a empty dictionary
        #         runningDict = module.updateModulesParameters(
        #             runningDict, module_key=None
        #         )  # call updateParameterDict function that updates the runningDict with the modules parameters

        #     for (
        #         current_key,  # iterate through the modules
        #         mod,
        #     ) in (
        #         module.getModules()
        #     ):  # for the module key and module value in the module dictionary
        #         if parent_key:
        #             current_key = f"{parent_key}.{current_key}"  # the module_key is created as an f string interpolation with the parent.module if the parent key is exists
        #         else:
        #             current_key  # else there is no parent so current_key is default
        #         runningDict = mod.updateModulesParameters(
        #             runningDict, current_key
        #         )  # update the parameter dict passing in the runningDict and the new module_key
        #         if (
        #             mod.isChild()
        #         ):  # If the module has a child call the recursive function
        #             helperRecursive(
        #                 mod, runningDict=runningDict, parent_key=current_key
        #             )  # call this function to go through steps again for child

        #     return runningDict  # return the final dict

        # parameters_dict = helperRecursive(
        #     self
        # )  # call the helper function to get the named_parameters dict
        # list_parameters = [
        #     (k, v) for k, v in parameters_dict.items()
        # ]  # convert dict into list of tuples
        # return list_parameters  # return list_parameters

    def parameters(self):
        "Enumerate over all the parameters of this module and its descendents."
        # return self.named_parameters().values()
        # parameters = []  # create empty list
        # for values in self.named_parameters():  # go through all the named parameters
        #     parameters.append(values[1])  # get the second value (parameter value)
        # return parameters  # return the parameters list
        return [x[1] for x in self.named_parameters()]

    def add_parameter(self, k, v):
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k (str): Local name of the parameter.
            v (value): Value for the parameter.

        Returns:
            Parameter: Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        assert False, "Not Implemented"

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x=None, name=None):
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x):
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)
