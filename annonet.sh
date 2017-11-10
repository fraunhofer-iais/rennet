#! /bin/bash
set -e
# check if RENNET_ROOT is defined
if [[ -z "${RENNET_ROOT}" ]]; then
    if [ ! -d "$PWD/rennet/utils" ]; then
        echo "Please run the script from root of the rennet repo (containing annonet.sh)"
        echo "or, export the environment variable named RENNET_ROOT pointing to the root of the repo."
        exit 1
    else
        RENNET_ROOT="$PWD"
    fi
fi
export RENNET_ROOT

# check if the virtual environment RENNET_ENV is setup
export RENNET_ENV="$RENNET_ROOT/.rennet_env"

if [ ! -d $RENNET_ENV ]; then
    echo ""
    echo "Setting up a local python virtual environment."
    echo "This requires an Internet connection, and can take some time."
    echo ""
    echo "This setup will only occur the first time the script is run"
    echo "or, if the virtual environment at $RENNET_ENV was deleted"
    echo ""

    # get user's blessings
    read -n1 -rsp "Press any key to continue or CTRL+C to exit" key
    echo ""
    source $RENNET_ROOT/.setup.rennet_env.local.sh
fi
source "$RENNET_ENV/bin/activate"
export PYTHONPATH=$RENNET_ROOT:$PYTHONPATH

# run the annonet, passing it all the args
KERAS_BACKEND=theano python rennet/annonet.py "$@"

# deactivate the virtual environment when all is done
deactivate
