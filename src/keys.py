"""config/run/<run_id>.yaml のキー。

キーは「その値を消費する関数の引数名」（module.Class.method.arg）で、record の params にも同じキーで載る。
フックはその関数の引数を観測し、gate が yaml の値と比べるので、値は変換せずにその引数へ渡す。
上流に渡らない設定（seed、workers、閾値）は src/ の関数の引数で受ける（エントリモジュールの関数は __main__ で記録されるので置かない）。
"""

# ---------------------------------------------------------------- run 全体（src/runner.py、src/preprocess.py）
METHOD = "src.runner.run_all.method"
WORKERS = "src.runner.run_all.workers"
INSTANCE_TIMEOUT = "src.runner.run_all.instance_timeout"
STALL_TIMEOUT = "src.runner.run_all.stall_timeout"
SPLIT = "src.preprocess.instances.split"
N_INSTANCES = "src.preprocess.instances.n_instances"

# ---------------------------------------------------------------- autoscilab: 上流の引数
ENSEMBLE_MODEL = "autoscilab.llm.ensemble_client.EnsembleClient.__init__.model"
ENSEMBLE_BASE_URL = "autoscilab.llm.ensemble_client.EnsembleClient.__init__.base_url"
ENSEMBLE_K = "autoscilab.llm.ensemble_client.EnsembleClient.__init__.k"
ENSEMBLE_TEMPERATURE = "autoscilab.llm.ensemble_client.EnsembleClient.__init__.temperature"
ENSEMBLE_MAX_TOKENS = "autoscilab.llm.ensemble_client.EnsembleClient.__init__.max_tokens"
LARGE_MODEL = "autoscilab.llm.client.LLMClient.__init__.model"
LARGE_BASE_URL = "autoscilab.llm.client.LLMClient.__init__.base_url"
LARGE_MAX_COMPLETION_TOKENS = "autoscilab.llm.client.LLMClient.__init__.max_completion_tokens"
AGENT_ADAPTIVE = "autoscilab.llm.ensemble_agent.EnsembleAgent.__init__.adaptive"
AGENT_K_MAX = "autoscilab.llm.ensemble_agent.EnsembleAgent.__init__.k_max"
AGENT_STABILITY_THRESHOLD = "autoscilab.llm.ensemble_agent.EnsembleAgent.__init__.stability_threshold"
TOTAL_BUDGET = "autoscilab.llm.ensemble_agent.EnsembleAgent.run_round.total_budget"
N_CANDIDATES = "autoscilab.al.falsification.FalsificationSelector.__init__.n_candidates"
DIVERSITY_WEIGHT = "autoscilab.al.falsification.FalsificationSelector.__init__.diversity_weight"
SELECT_N = "autoscilab.al.falsification.FalsificationSelector.select.n"

# ---------------------------------------------------------------- autoscilab: src/model.py の関数の引数
ACQUISITION = "src.model.TrajectoryDistribution.over.acquisition"
N_BOOTSTRAP = "src.model.bootstrap_confidence.n_bootstrap"
MIN_R2 = "src.model.bootstrap_confidence.min_r2"
CONFIDENCE_THRESHOLD = "src.model.refine_gate.confidence_threshold"
SEED = "src.model.make_rng.seed"
BOUNDS_FACTOR = "src.model.System.allowed_bounds.factor"
SIM_TIMEOUT = "src.model.set_sim_timeouts.sim_timeout"
SIM_TIMEOUT_PER_JOB = "src.model.set_sim_timeouts.sim_timeout_per_job"

# ---------------------------------------------------------------- react: 上流（SciGym、openai）の引数
REACT_MAX_ITERATIONS = "scigym.controller.Controller.__init__.max_iterations"
REACT_EVAL_DEBUG_ROUNDS = "scigym.controller.Controller.__init__.eval_debug_rounds"
REACT_TEMPERATURE = "scigym.controller.Controller.__init__.temperature"
REACT_MODEL = "scigym.api.api.LLM.__init__.model_name"
REACT_LLM_TEMPERATURE = "scigym.api.api.LLM.__init__.temperature"
REACT_MAX_LENGTH = "scigym.api.api.LLM.__init__.max_length"
REACT_BASE_URL = "openai._client.OpenAI.__init__.base_url"
