###################################################
# Author: Ant Bilic                               #
# Since: Feb 22, 2024                             #
# Copyright: DAFF Hitchhikers working group       #
# Version: N/A                                    #
# Maintainer: Ante Bilic                          #
# Email: ante.bilic.mr@gmail.com                  #
# Status: N/A                                     #
###################################################

"""
Creates a Config object from the input YAML file
"""

from pathlib import Path
import yaml
from loguru import logger

class Config:
    """
    Class representing the global configuration

    Parameters:
    ----------
    yaml_file : string
        The path to the yaml configuration file
    """

    def __init__(self, yaml_file: str | Path) -> None:
        with open(yaml_file, "r", encoding="utf-8") as fin:
            conf_dic = yaml.safe_load(fin)

        self.name = Path(yaml_file).name.rsplit(".", 1)[0]

        if "inputs" in conf_dic:
            try:
                self.data_dir = Path(conf_dic['inputs']['data']['data_dir'])
                self.ext_dir = self.data_dir / conf_dic['inputs']['data']['ext_dir']
                self.cpt_top = conf_dic['inputs']['data']['top_cpt']
                self.cal_med_approach_rate = conf_dic['inputs']['cal_medium_approach_rate']
                self.cal_med_fraction = conf_dic['inputs']['cal_medium_fraction']
                self.tot_containers = conf_dic['inputs']['virtconts']['tot_containers']
                self.vcon_dir = Path(conf_dic['inputs']['virtconts']['vcon_dir'])
                self.num_label = conf_dic['inputs']['virtconts']['num_label']
                self.tot_importers = conf_dic['inputs']['virtconts']['tot_importers']
                if isinstance(self.tot_importers, list) and \
                all(dic.get(_k) >= 1 for dic in self.tot_importers for _k in dic if _k == "volume"):
                    self.tot_containers = sum(dic.get(_k) for dic in self.tot_importers
                                              for _k in dic if _k == "volume")
            except Exception as _e:
                raise InputConfigError(conf_dic['inputs']['data']['data_dir'],
                                       "Cannot read the inputs block") from _e
            self.vcon_file = self.vcon_dir /\
                f"{self.ext_dir.name}_containers_external_with_medium_{self.num_label}.csv"


        if "work" in conf_dic:
            try:
                self.workdir = Path(conf_dic["work"]["w_dir"])
                self.old_wdir_single = self.workdir /\
                    conf_dic["work"]["single_dir"] / conf_dic["work"]["old"]
                self.new_wdir_single = self.workdir /\
                    conf_dic["work"]["single_dir"] / conf_dic["work"]["new"]
                self.old_wdir_multi = self.workdir /\
                    conf_dic["work"]["multi_dir"] / conf_dic["work"]["old"]
                self.new_wdir_multi = self.workdir /\
                    conf_dic["work"]["multi_dir"] / conf_dic["work"]["new"]
                self.visdir = Path(conf_dic["work"]["vis_dir"])
                self.old_vdir_single = self.visdir /\
                    conf_dic["work"]["single_dir"] / conf_dic["work"]["old"]
                self.new_vdir_single = self.visdir /\
                    conf_dic["work"]["single_dir"] / conf_dic["work"]["new"]
                self.old_vdir_multi = self.visdir /\
                    conf_dic["work"]["multi_dir"] / conf_dic["work"]["old"]
                self.new_vdir_multi = self.visdir /\
                    conf_dic["work"]["multi_dir"] / conf_dic["work"]["new"]
                self.sim_total = conf_dic["work"]["sim_total"]
            except Exception as _e:
                raise InputConfigError(conf_dic['work']['w_dir'],
                                       "Cannot read the work block") from _e

        if "sensi_rates" in conf_dic:
            try:
                self.fun_params = conf_dic["sensi_rates"]
                self.fun_params.update(conf_dic["to_save"])
            except Exception as _e:
                raise InputConfigError(conf_dic['sensi_rates']['container_filename'],
                                       "Cannot read the function parameters block") from _e
        if "port_monitor" in conf_dic:
            self.fun_params.update({"port_monitor": conf_dic["port_monitor"]})
            if self.fun_params["port_monitor"]:
                self.fun_params.update({"monitor_column": conf_dic["monitor_column"]})

        logger.info(f"\nConfig parameters:\n{dir(self)}")


class InputConfigError(Exception):
    """
    Exception raised for errors in the input YAML file

    Parameters:
        conf_par : string
            the parameter which caused the error
        message : string
            the explanation of the error
    """

    def __init__(self, conf_par: str, message: str="Config parameter(s) invalid") -> None:
        self.conf_par = conf_par
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}: {self.conf_par}"
