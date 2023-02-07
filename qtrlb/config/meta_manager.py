
class MetaManager:
    """ A container for all managers to be accessible through one object.
        Each of their name will become an attribute of MetaManager.
    
        Example of manager_dict:
        manager_dict = {'variables': varman,
                        'DAC': dacman}
    """
    def __init__(self, manager_dict):
        self.manager_dict = manager_dict
        
        for manager_name, manager in self.manager_dict:
            self.__setattr__(manager_name, manager)
      
    
    def load(self):
        """
        Call all managers' load method, VariableManager first.
        """
        self.variables.load()
        for manager in self.manager_dict.values():
            manager.load()  
        # It's fine we load VariableManager twice. It's fast.
        
        
    def save(self):
        for manager in self.manager_dict.values():
            manager.save()
        
        
        