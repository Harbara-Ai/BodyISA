_LOCAL_EVALUATION_MARKER=object()

class LocalEvaluationPermit:
    """In-process benchmark permit, intentionally absent from instruction input."""
    __slots__=('run_id','_marker')
    def __init__(self,run_id,marker):
        if marker is not _LOCAL_EVALUATION_MARKER:raise RuntimeError('LOCAL_EVALUATION_ONLY')
        self.run_id=run_id
        self._marker=marker


def issue_local_evaluation_permit(run_id):
    if not isinstance(run_id,str) or not run_id:raise ValueError('run_id required')
    return LocalEvaluationPermit(run_id,_LOCAL_EVALUATION_MARKER)


