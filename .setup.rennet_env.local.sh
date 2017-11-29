#! /bin/bash
# Build a python virtual environment in the repo's root
if [ ! -d $RENNET_ENV ]
then
    mkdir $RENNET_ENV
fi

VENV="$RENNET_ROOT/.virtualenv-setup"
if [ ! -d $VENV ]
then
    git clone -b 15.1.0 --depth=1 https://github.com/pypa/virtualenv.git $VENV
fi

/usr/bin/python "$VENV/virtualenv.py" $RENNET_ENV
source "$RENNET_ENV/bin/activate"

echo ""
echo ""
echo "Installing Packages"
echo ""
echo ""
pip install pip --upgrade
pip install six

pip install -r "$RENNET_ROOT/requirements/base.txt"

echo ""
echo ""
echo "All Packages were installed. The environment is ready."
echo ""
echo ""
rm -rf $VENV
deactivate
