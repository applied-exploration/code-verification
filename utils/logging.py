from rich.console import Console
from rich.table import Table
from typing import List
from collections import defaultdict


class RichConsole:
    def __init__(self):
        self.console: Console = Console()
        self.tables: dict = defaultdict(Table)

    def new_table(self, table_name: str, title: str):
        self.tables[table_name] = Table(
            show_header=True,
            header_style="bold magenta",
            title=title,
            width=self.console.width,
        )

    def define_columns(self, table_name: str, column_names: List[str]):
        table = self.tables[table_name]
        for column_name in column_names:
            table.add_column(column_name, style="dim", width=12)

    def add_row_list_to_table(self, table_name: str, rows: List):
        table = self.tables[table_name]

        for row in rows:
            table.add_row(*row)

    def display(self, table_name):
        self.console.print(self.tables[table_name])
