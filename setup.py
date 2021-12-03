from setuptools import setup

options=dict( name='bigbets',
    version='1.0.0',
    packages=['bigbets'],
    license='GPLv3+',
    author='William H. Weir',
    provides=['bigbets'],
    author_email='wweir@med.unc.edu',
    description='bipartite graph rewiring using the TCGA data.',
    zip_safe=False,
    classifiers=["Programming Language :: Python :: 3.6",
                 "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                 "Topic :: Scientific/Engineering :: Information Analysis",
                 ],
    install_requires=['numpy','scipy','sklearn']
)

setup(**options)
