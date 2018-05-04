"""
Logs for environment rollouts.
"""

import os
import time

import gym
import pandas

# pylint: disable=E0202


class LoggedEnv(gym.Wrapper):
    """
    An environment that logs episodes to a file.

    Logs are CSV files with three columns:
      r: episode reward
      l: episode length (timesteps)
      t: timestamp of episode end, relative to log start.
    """

    def __init__(self, env, log_path, use_locking=False):
        """
        Create a logged environment.

        Args:
          env: the environment to wrap.
          log_path: path to output CSV file.
          use_locking: use UNIX file locking to allow
            multiple processes to write to one log.

        If the log file already exists, it is appended to.
        """
        super(LoggedEnv, self).__init__(env)
        self._start_time = time.time()
        self._use_locking = use_locking
        self._file_desc = os.open(log_path, os.O_RDWR | os.O_CREAT)
        self._file = os.fdopen(self._file_desc, 'r+t')
        try:
            self._initialize_file()
        except:
            self._file.close()
            raise
        self._cur_timesteps = 0
        self._cur_reward = 0

    def reset(self, **kwargs):
        self._cur_timesteps = 0
        self._cur_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._cur_reward += rew
        self._cur_timesteps += 1
        if done:
            self._write_entry()
            self._cur_timesteps = 0
            self._cur_reward = 0
        return obs, rew, done, info

    def close(self):
        self.env.close()
        self._file.close()

    def _set_locked(self, locked):
        if not self._use_locking:
            return
        import fcntl
        if locked:
            fcntl.lockf(self._file_desc, fcntl.LOCK_EX)
        else:
            fcntl.lockf(self._file_desc, fcntl.LOCK_UN)

    def _write_entry(self):
        """
        Write the latest episode record.
        """
        info = {k: [v] for k, v in self._episode_info().items()}
        data = pandas.DataFrame(info).to_csv(columns=self._columns, header=False, index=False)
        self._set_locked(True)
        try:
            self._file.seek(0, os.SEEK_END)
            self._file.write(data)
            self._file.flush()
        finally:
            self._set_locked(False)

    def _episode_info(self):
        """
        Get a dict of info about the latest episode.
        """
        return {
            'r': self._cur_reward,
            'l': self._cur_timesteps,
            't': time.time() - self._start_time
        }

    def _initialize_file(self):
        """
        Setup the initial file or read existing meta-data
        from it.
        """
        self._set_locked(True)
        self._file.seek(0, os.SEEK_END)
        if self._file.tell() > 0:
            self._file.seek(0, os.SEEK_SET)
            contents = pandas.read_csv(self._file)
            if len(contents) > 0:
                self._start_time = time.time() - max(contents['t'])
        else:
            self._write_header()
        self._set_locked(False)

    def _write_header(self):
        """
        Write the CSV header.
        """
        self._file.write(','.join(self._columns) + '\n')
        self._file.flush()

    @property
    def _columns(self):
        """
        Get the columns for the CSV file.
        """
        return ['r', 'l', 't']
