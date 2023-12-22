from india_insurance.entity import artifact_entity,config_entity
from india_insurance.exception import india_insuranceException
from india_insurance.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
import os,sys 
import pandas as pd
from india_insurance import utils
import numpy as np
from india_insurance.config import TARGET_COLUMN