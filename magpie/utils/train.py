import argparse

import argconf

import magpie.model as model


def main():
    description = "Trains a Magpie model."
    epilog = "Usage:\npython -m magpie.utils.train --config confs/cae_model_config.json "\
        "--options confs/options.json > cae_train_log"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("--config", type=str, default="confs/cae_model_config.json")
    parser.add_argument("--options", type=str, default="confs/options.json")
    args, _ = parser.parse_known_args()

    option_dict = argconf.options_from_json(args.options)
    config = argconf.config_from_json(args.config)
    config = argconf.parse_args(option_dict, config=config)

    trainer_cls = model.find_trainer(config["trainer_type"])
    trainer = trainer_cls(config)
    trainer.train()


if __name__ == "__main__":
    main()