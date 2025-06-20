from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput  # python-call-graph

with PyCallGraph(output=GraphvizOutput()):
    main_user.main()
