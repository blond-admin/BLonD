from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput  # python-call-graph
import main_user

with PyCallGraph(output=GraphvizOutput()):
    main_user.main()
