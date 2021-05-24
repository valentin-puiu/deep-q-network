class TargetNetworkUpdater(object):
    """Actualizeaza reteaua tinta cu cea principala"""
    def __init__(self, main_dqn_vars, target_dqn_vars):
        """
            main_dqn_vars: variabilele retelei principale
            target_dqn_vars: variabilele retelei tinta
        """
        self.main_dqn_vars = main_dqn_vars
        self.target_dqn_vars = target_dqn_vars

    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops
            
    def __call__(self, sess):
        """
        sess: sesiunea de tensorflow
        Aloca valorile parametrilor retelei principale la reteaua tinta
        """
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)