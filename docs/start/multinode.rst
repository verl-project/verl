Multinode Training
==================

Last updated: 06/10/2025.

.. _wuxibin89: https://github.com/wuxibin89

Author: `Xibin Wu <https://github.com/wuxibin89>`_, `Yusheng Su <https://yushengsu-thu.github.io/>`_.

Option 1: Launch Manually
------------------------------

Set up multinode ray cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Start head node with ``ray start --head --dashboard-host=0.0.0.0``, there're 2 address you should care about:

- GCS address: ``ray start --address=<address>``, where worker node should connect to.
- Dashboard address: ``<address>:8265``, where you should submit job to the cluster.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/head.png?raw=true

2. Start worker node with ``ray start --address=<address>`` you get above.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/worker.png?raw=true

3. Now you should see the cluster have 2 nodes with ``ray status``.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/status.png?raw=true

4. Additionally, you can access dashboard in the browser with the address you get above. 

*Firewall rules maybe need configure to access the dashboard, if there's any trouble, please contact your network administrator.*

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/overview.png?raw=true

Submit job to ray cluster
~~~~~~~~~~~~~~~~~~~~~~~~~
1. Submit ray job to cluster with the dashboard address you get above.

.. code-block:: bash

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        --no-wait \
        -- \
        python3 -m verl.trainer.main_ppo \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        ...

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/submit.png?raw=true

2. Then you can check the job status with the following commands:

- ray job list: list all jobs submitted to the cluster.
- ray job logs <Submission ID>: query the logs of the job.
- ray job status <Submission ID>: query the status of the job.
- ray job stop <Submission ID>: request the job to be stopped.
- ray job list | grep submission_id | grep JobStatus | grep RUNNING | grep -oP 'raysubmit_[^'\''"]+' | head -n 1: get the latest job submission ID of the running job.
- ray job logs <Submission ID> --follow: added ``--follow`` parameter to ray job logs command to enable continuous log streaming.

3. You can also access driver/task/actor logs in ``/tmp/ray/session_latest/logs/``, driver log is ``job-driver-raysubmit_<Submission ID>.log``.

4. We strongly recommend you to view job detail from dashboard in multinode training, because it provide more structure way to view the job information.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/job.png?raw=true
.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/job_detail.png?raw=true

Option 2: Launch via SkyPilot on Kubernetes or clouds
------------------------------------------------------

.. note::
   Ready-to-use SkyPilot example configurations are available in the `examples/skypilot/ <https://github.com/verl-project/verl/tree/main/examples/skypilot>`_ directory:
   
   - ``verl-ppo.yaml`` - PPO training with GSM8K dataset
   - ``verl-grpo.yaml`` - GRPO training with MATH dataset  
   - ``verl-multiturn-tools.yaml`` - Multi-turn tool usage training
   
   See the `SkyPilot examples README <https://github.com/verl-project/verl/tree/main/examples/skypilot>`_ for detailed usage instructions.

Step 1: Setup SkyPilot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SkyPilot can support different clouds, here we use GCP as example. `install skypilot <https://docs.skypilot.co/en/latest/getting-started/installation.html>`_

.. code-block:: bash

    conda create -y -n sky python=3.10
    conda activate sky
    pip install "skypilot[gcp]"

    conda install -c conda-forge google-cloud-sdk
    gcloud init

    # Run this if you don't have a credential file.
    # This will generate ~/.config/gcloud/application_default_credentials.json.
    gcloud auth application-default login

    # Check if the GCP credential is correctly setup.
    sky check gcp

.. image:: https://github.com/yottalabsai/open-source/blob/main/static/verl/setup_skypilot.png?raw=true

Step 2: Prepare dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/verl-project/verl.git
   cd examples/data_preprocess
   python3 gsm8k.py --local_save_dir ~/data/gsm8k


Step 3: Submit a job with SkyPilot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Create a SkyPilot YAML ``verl-cluster.yml`` with the following content:

.. parsed-literal:: workdir: .  will sync all the data in the current dir to the remote cluster.

.. code-block:: yaml

   resources:
     accelerators: L4:1 # every node has 1 L4 GPU
     image_id: docker:verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.0-fa2.7.4
     memory: 64+        # every node has 64 GB memory
     ports: 8265        # expose port for ray dashboard

   num_nodes: 2         # cluster size

   # --------------- Work Directory Synchronization (workdir) ---------------
   # Defines the local working directory to be synchronized to the remote cluster.
   # Here, '.' means synchronizing the directory where the sky submit command is currently run.
   workdir: .

   # --------------- (secrets) ---------------
   secrets:
     ## your wandb api key ##
     WANDB_API_KEY: null

   # --------------- File Mounts/Data Upload (file_mounts) ---------------
   # If your dataset (gsm8k folder) is local, it needs to be uploaded to the remote cluster.
   file_mounts:
     # Remote path (relative to remote user's home directory): Local path
     # /remote/dir1/file: /local/dir1/file
     data/gsm8k: ~/data/gsm8k

   # --------------- Environment Setup (setup) ---------------
   # Commands run on each node of the remote cluster to set up the environment (e.g., install dependencies). These are run directly inside Docker.
   setup: |
     rm -rf verl
     git clone https://github.com/verl-project/verl.git
     cd verl
     pip3 install -v -e .[vllm]

   # --------------- Run Command (run) ---------------
   # The actual task commands to be executed on the remote cluster.
   # This script will first start the Ray cluster (different ray start commands are executed on Head and Worker nodes).
   # Then, your training script will only be run on the Head node (SKYPILOT_NODE_RANK == 0).
   run: |
     # Get the Head node's IP and total number of nodes (environment variables injected by SkyPilot).
     head_ip=`echo "$SKYPILOT_NODE_IPS" | head -n1`
     num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l` # Here num_nodes should be equal to 2.

     # login wandb
     python3 -c "import wandb; wandb.login(relogin=True, key='$WANDB_API_KEY')"

     # Start Ray based on node role (Head=0, Worker>0).
     # This logic is a standard Ray cluster startup script.
     if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
       # Head node starts Ray Head.
       echo "Starting Ray head node..."
       # Check if a Ray Head is already running to avoid duplicate starts.
       ps aux | grep ray | grep 6379 &> /dev/null ||  ray start --head --disable-usage-stats \
             --port=6379 \
             --dashboard-host=0.0.0.0 \
             --dashboard-port=8265

       # Wait for all worker nodes to join the cluster.
       while [ $(ray nodes | grep NODE_ID | wc -l) -lt $num_nodes ]; do
         echo "Waiting for all nodes to join... ($(ray nodes | grep NODE_ID | wc -l)/$num_nodes)"
         sleep 5
       done

       # Head node executes the training script.
       echo "Executing training script on head node..."

       python3 -m verl.trainer.main_ppo \
        data.train_files=data/gsm8k/train.parquet \
        data.val_files=data/gsm8k/test.parquet \
        data.train_batch_size=256 \
        data.max_prompt_length=512 \
        data.max_response_length=256 \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
        critic.ppo_micro_batch_size_per_gpu=4 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['console','wandb'] \
        trainer.val_before_train=False \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=2 \
        trainer.save_freq=20 \
        trainer.test_freq=20 \
        trainer.total_epochs=2 \
        trainer.project_name=verl_examples \
        trainer.experiment_name=experiment_name_gsm8k

     else
       # Wait for Ray Head to start.
       sleep 10 # Increase waiting time to ensure Head finishes starting.
       # Worker node starts Ray Worker.
       echo "Starting Ray worker node..."

       # Check if a Ray Worker is already running to avoid duplicate starts.
       ps aux | grep ray | grep $head_ip:6379 &> /dev/null || ray start --address $head_ip:6379 --disable-usage-stats

       # Add sleep to after `ray start` to give ray enough time to daemonize
       sleep 5 # Ensure Worker successfully connects to Head.
     fi

     # No commands are added to the Worker node here; the Worker's main task is to start Ray and wait for the Head node to assign tasks.
     echo "Node setup and Ray start script finished for rank $SKYPILOT_NODE_RANK."


.. code-block:: bash

    export WANDB_API_KEY=<your-wandb-api-key>
    sky launch -c verl --secret WANDB_API_KEY verl-cluster.yml

.. image:: https://github.com/yottalabsai/open-source/blob/main/static/verl/running_job.png?raw=true
.. image:: https://github.com/yottalabsai/open-source/blob/main/static/verl/running_job_1.png?raw=true
.. image:: https://github.com/yottalabsai/open-source/blob/main/static/verl/finished.png?raw=true

**Check the cluster on GCP**

.. image:: https://github.com/yottalabsai/open-source/blob/main/static/verl/gcp_instances.png?raw=true

**Check Ray Dashboard**

We can see the cluster on the RAY Dashboard with the GCP head node:

```console
$ sky status --endpoint 8265 verl
1.2.3.4:8265
```

.. image:: https://github.com/yottalabsai/open-source/blob/main/static/verl/ray_dashboard_overview.png?raw=true
.. image:: https://github.com/yottalabsai/open-source/blob/main/static/verl/ray_dashboard_jobs.png?raw=true
.. image:: https://github.com/yottalabsai/open-source/blob/main/static/verl/ray_dashboard_cluster.png?raw=true


**Check the checkpoint of model**

.. code-block:: bash

    # login the head node
    ssh verl
    # The global step will vary. Find the correct path from the training logs.
    cd ~/sky_workdir/checkpoints/verl_examples/gsm8k/
    # Then list contents to find the checkpoint, e.g.:
    ls -R .

.. image:: https://github.com/yottalabsai/open-source/blob/main/static/verl/saved_model.png?raw=true


Option 3: Launch via Slurm
------------------------------

Ray provides users with `this <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ official
tutorial to start a Ray cluster on top of Slurm. We have verified the :doc:`GSM8K example<../examples/gsm8k_example>`
on a Slurm cluster under a multi-node setting with the following steps.

1. [Optional] If your cluster support `Apptainer or Singularity <https://apptainer.org/docs/user/main/>`_ and you wish
to use it, convert verl's Docker image to an Apptainer image. Alternatively, set up the environment with the package
manager available on your cluster or use other container runtimes (e.g. through `Slurm's OCI support <https://slurm.schedmd.com/containers.html>`_) available to you.

.. code:: bash

    apptainer pull /your/dest/dir/vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3.sif docker://verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3

2. Follow :doc:`GSM8K example<../examples/gsm8k_example>` to prepare the dataset and model checkpoints.

3. Modify `examples/slurm/ray_on_slurm.slurm <https://github.com/verl-project/verl/blob/main/examples/slurm/ray_on_slurm.slurm>`_ with your cluster's own information.

4. Submit the job script to the Slurm cluster with `sbatch`.

Please note that Slurm cluster setup may vary. If you encounter any issues, please refer to Ray's
`Slurm user guide <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ for common caveats.

If you changed Slurm resource specifications, please make sure to update the environment variables in the job script if necessary.


Option 4: Launch via dstack
------------------------------

`dstackai/dstack <https://github.com/dstackai/dstack>`_ is an open-source container orchestrator that simplifies distributed training across cloud providers and on-premises environments
without the need to use K8S or Slurm.

Prerequisite
~~~~~~~~~~~~
Once dstack is `installed <https://dstack.ai/docs/installation>`_, initialize the directory as a repo with ``dstack init``. 

.. code-block:: bash

    mkdir myproject && cd myproject
    dstack init

**Create a fleet**

Before submitting distributed training jobs, create a `dstack` `fleet <https://dstack.ai/docs/concepts/fleets>`_.

Run a Ray cluster task
~~~~~~~~~~~~~~~~~~~~~~

Once the fleet is created, define a Ray cluster task, e.g. in ``ray-cluster.dstack.yml``:

.. code-block:: yaml

    type: task
    name: ray-verl-cluster

    nodes: 2

    env:
        - WANDB_API_KEY
        - PYTHONUNBUFFERED=1
        - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    
    image: verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2
    commands:
        - git clone https://github.com/verl-project/verl
        - cd verl
        - pip install --no-deps -e .
        - pip install hf_transfer hf_xet
        - |
        if [ $DSTACK_NODE_RANK = 0 ]; then
            python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
            python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-7B-Instruct')" 
            ray start --head --port=6379;
        else
            ray start --address=$DSTACK_MASTER_NODE_IP:6379
        fi

    # Expose Ray dashboard port
    ports:
        - 8265

    resources:
        gpu: 80GB:8
        shm_size: 128GB

    # Save checkpoints on the instance
    volumes:
        - /checkpoints:/checkpoints

Now, if you run this task via `dstack apply`, it will automatically forward the Ray's dashboard port to `localhost:8265`.

.. code-block:: bash

    dstack apply -f ray-cluster.dstack.yml

As long as the `dstack apply` is attached, you can use `localhost:8265` to submit Ray jobs for execution

Submit Ray jobs
~~~~~~~~~~~~~~~

Before you can submit Ray jobs, ensure to install `ray` locally:
   
.. code-block:: shell

    pip install ray

Now you can submit the training job to the Ray cluster which is available at ``localhost:8265``:
   
.. code-block:: shell

    $ RAY_ADDRESS=http://localhost:8265
    $ ray job submit \
        -- python3 -m verl.trainer.main_ppo \
        data.train_files=/root/data/gsm8k/train.parquet \
        data.val_files=/root/data/gsm8k/test.parquet \
        data.train_batch_size=256 \
        data.max_prompt_length=512 \
        data.max_response_length=256 \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=Qwen/Qwen2.5-7B-Instruct \
        critic.ppo_micro_batch_size_per_gpu=4 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.project_name=ppo_training \
        trainer.experiment_name=qwen-2.5-7B \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.default_local_dir=/checkpoints \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log \
        trainer.resume_mode=disable


For more details on how `dstack` works, check out its `documentation <https://dstack.ai/docs>`_.

How to debug?
---------------------


Ray Distributed Debugger VSCode Extension (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Starting with Ray 2.39, Anyscale has introduced the `Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`_ VSCode extension. Follow the extension’s installation instructions, then add your cluster using the dashboard URL you obtained earlier.

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/debugger.png?raw=true
      :alt: Ray Distributed Debugger VSCode extension screenshot

2. Prerequisites.

   Ensure the following are installed (see the extension README for more detail):

   - Visual Studio Code  
   - `ray[default]` >= 2.9.1  
   - `debugpy` >= 1.8.0  

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/c7098b755ff689859837773a916c857.png?raw=true
      :alt: VSCode with Ray prerequisites

3. Environment Variables.

   To enable post‑mortem debugging, set:

   .. code-block:: bash

      export RAY_DEBUG_POST_MORTEM=1

   .. admonition:: Note
      :class: important

      Be sure to remove any legacy flags before starting Ray:

      - `RAY_DEBUG=legacy`  
      - `--ray-debugger-external`

4. Configuring BreakpointsSet up breakpoint() in your code, and submit job to cluster. Then the extension will show the breakpoint information.


   1. Insert `breakpoint()` calls into your remote functions.  
   2. Submit your job to the cluster.  

   The extension will detect active breakpoints and display them in VSCode.

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: Detected breakpoint in VSCode

   **Note:** Breakpoints are only supported inside functions decorated with `@ray.remote`.

5. Launching the Debugger.

   Run your job directly from the command line (do not use a `launch.json`):

   .. code-block:: bash

      python job.py

6. Attaching to a Breakpoint.

 Once the process hits the first `breakpoint()`, click the Ray Distributed Debugger icon in the VSCode sidebar to attach the debugger.

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: Attaching VSCode debugger to Ray process

7. Debugging With Multiple breakpoint().

   For each subsequent task, first disconnect the current debugger session, then click the extension icon again to attach to the next breakpoint.

   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/6e83c910a62c82fecb89c6619e001cd.png?raw=true
      :alt: Disconnecting and reconnecting the debugger

Legacy Ray Debugger
~~~~~~~~~~~~~~~~~~~
1. Ray has a builtin legacy `debugger <https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html>`_ that allows you to debug your distributed applications. To enable debugger, start ray cluster with ``RAY_DEBUG=legacy`` and ``--ray-debugger-external``.

.. code-block:: bash

    # start head node
    RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
    # start worker node
    RAY_DEBUG=legacy ray start --address='10.124.46.192:6379' --ray-debugger-external

2. Set up breakpoint in your code, and submit job to cluster. Then run ``ray debug`` to wait breakpoint:

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/legacy.png?raw=true


Multi-node training on AMD clusters
---------------------------------------------------------------------------------------

If you want to run multi-node training with slurm with Docker/Podman container on AMD Cluster, you can use the following script. 

If you encounter any issues in using AMD GPUs running verl, please contact `Yusheng Su <https://yushengsu-thu.github.io/>`_.

.. note::
    1. You need to use ``podman`` or ``docker`` in the following script. We will release the apptainer script later.
    2. If you want to use ``podman``, you just replace ``docker`` with ``podman`` in the following script.

The script includes the following steps:

1. SLURM Configuration
2. Environment Setup
3. Docker/Podman Container Setup
4. Ray Cluster Initialization
5. Data Preprocessing
6. Model Setup
7. Training Launch


slurm_script.sh
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    #!/bin/bash

    #SBATCH --job-name=verl-ray-on-slurm
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=2
    #SBATCH --mem=200G
    #SBATCH --time=30-00:00:00
    #SBATCH --gpus-per-node=8
    #SBATCH --cpus-per-task=28
    #SBATCH --output=../verl_log/slurm-%j.out
    #SBATCH --error=../verl_log/slurm-%j.err
    #SBATCH --nodelist=gpu-[0,1]


    # load necessary modules
    ### Run this setup
    # [Cluster]: Use docker
    # docker pull docker.io/rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4


    ##########################################################################
    ###The following setting should be set in different project and cluster###
    ##########################################################################

    ### Project
    CONTAINER_NAME="multinode_verl_training"
    IMG="verl.rocm"
    DOCKERFILE="docker/Dockerfile.rocm"
    # echo $PWD
    verl_workdir="${HOME}/projects/verl_upstream"
    export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface"
    export HF_HOME=$TRANSFORMERS_CACHE

    ### Cluster Network Setting
    export NCCL_DEBUG=TRACE
    export GPU_MAX_HW_QUEUES=2
    export TORCH_NCCL_HIGH_PRIORITY=1
    export NCCL_CHECKS_DISABLE=1
    # export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
    export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_PROTO=Simple
    export RCCL_MSCCL_ENABLE=0
    export TOKENIZERS_PARALLELISM=false
    export HSA_NO_SCRATCH_RECLAIM=1
    ##########################################################################

    ### For rocm and training script
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES


    # Build and launch the Docker container
    srun bash -c "
        # Exit on any error
        set -e 

        # Clean up dangling images (images with <none> tag)
        docker image prune -f

        # Need to pull the docker first
        docker pull docker.io/rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
        
        if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "${IMG}"; then
            echo \"Building ${IMG} image...\"
            docker build -f \"${DOCKERFILE}\" -t \"${IMG}\" .
        else
            echo \"${IMG} image already exists, skipping build\"
        fi

        # Removing old container if exists
        docker rm \"${CONTAINER_NAME}\" 2>/dev/null || true

        # Checking network devices
        ibdev2netdev

        # Launch the docker
        docker run --rm -d \
        -e HYDRA_FULL_ERROR=1 \
        -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
        -e ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        -e NCCL_DEBUG=${NCCL_DEBUG} \
        -e GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES} \
        -e TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY} \
        -e NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE} \
        -e NCCL_IB_HCA=${NCCL_IB_HCA} \
        -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
        -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC} \
        -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \
        -e NCCL_PROTO=${NCCL_PROTO} \
        -e RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE} \
        -e TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM} \
        -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM} \
        -e TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} \
        -e HF_HOME=${HF_HOME} \
        --network host \
        --device /dev/dri \
        --device /dev/kfd \
        --device /dev/infiniband \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --privileged \
        -v \${HOME}:\${HOME} \
        -v \${HOME}/.ssh:/root/.ssh \
        -w "${verl_workdir}" \
        --shm-size 128G \
        --name \"${CONTAINER_NAME}\" \
        \"${IMG}\" \
        tail -f /dev/null

        echo \"Container setup completed\"
    "
        # (Optional): If you do not want to root mode and require assign yuorself as the user
        # Please add `-e HOST_UID=$(id -u)` and `-e HOST_GID=$(id -g)` into the above docker launch script. 





    ### Ray launch the nodes before training

    # Getting the node names
    nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # if we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
        echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    # make sure we set environment variables before Ray initialization

    # Print out all env variables
    printenv

    echo "Starting HEAD at $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        docker exec "${CONTAINER_NAME}" \
            ray start --head --node-ip-address="$head_node_ip" --port=$port \
            --dashboard-port=8266 \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10

    # number of nodes other than the head node
    worker_num=$((SLURM_JOB_NUM_NODES - 1))

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Debug: Starting worker on node_i = ${node_i}"
        if [ -z "$node_i" ]; then
            echo "Error: Empty node name for worker $i"
            continue
        fi
        echo "Starting WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            docker exec "${CONTAINER_NAME}" \
                ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
        sleep 5
    done




    # Ray initlization test (See whether any error in the above execution)
    echo "Testing Ray initialization in the slurm nodes..."
    docker exec "${CONTAINER_NAME}" python3 -c '
    import ray
    try:
        ray.init(address="auto")
        print("\n=== Ray Cluster Status ===")
        print(f"Number of nodes: {len(ray.nodes())}")
        for node in ray.nodes():
            print("Node: {}, Status: {}".format(node["NodeManagerHostname"], node["Alive"]))
            # print(f"Node: {node}")
        ray.shutdown()
        print("Ray initialization successful!")
    except Exception as e:
        print(f"Ray initialization failed: {str(e)}")
    '
    echo "=== Ray test completed ==="
    ######



    # Run data preprocessing

    echo "Starting data preprocessing..."
    docker exec "${CONTAINER_NAME}" \
        python3 "examples/data_preprocess/gsm8k.py" "--local_save_dir" "../data/gsm8k"

    echo "Starting data preprocessing..."
    docker exec "${CONTAINER_NAME}" \
        python3 "examples/data_preprocess/math_dataset.py" "--local_dir" "../data/math"

    train_files="../data/gsm8k/train.parquet"
    val_files="../data/gsm8k/test.parquet"

    # Download and test model
    echo "Loading model..."
    docker exec "${CONTAINER_NAME}" \
        python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
    MODEL_PATH="Qwen/Qwen2-7B-Instruct"

    # Set model path after pipeline test
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"

    echo "== Data and model loading Done =="

    echo "Start to train..."

    docker exec "${CONTAINER_NAME}" \
        python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
    MODEL_PATH="Qwen/Qwen2-7B-Instruct"


    PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node" \
        docker exec "${CONTAINER_NAME}" \
        python3 -m verl.trainer.main_ppo \
        data.train_files=$train_files \
        data.val_files=$val_files \
        data.train_batch_size=1024 \
        data.max_prompt_length=1024 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=$MODEL_PATH \
        critic.model.enable_gradient_checkpointing=False \
        critic.ppo_micro_batch_size_per_gpu=8 \
        critic.fsdp.param_offload=False \
        critic.fsdp.optimizer_offload=False \
        algorithm.kl_ctrl.kl_coef=0.0001 \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='verl_example' \
        trainer.experiment_name='Qwen2.5-32B-Instruct_function_rm' \
        trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
        trainer.val_before_train=False \
        trainer.nnodes=${SLURM_NNODES} \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15


Run multi-node training with above slurm_script.sh
~~~~~~~~~~~~~~~~~~~~
Just sbatch your slurm_script.sh

.. code-block:: bash

    sbatch slurm_script.sh

