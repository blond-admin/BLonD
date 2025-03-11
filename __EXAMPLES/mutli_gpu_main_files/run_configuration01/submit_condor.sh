#!/bin/bash
# Shortcut to execute the simulation on HTCondor

condor_submit --verbose config_condor.sub && watch condor_q
